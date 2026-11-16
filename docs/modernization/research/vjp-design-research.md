# VJP Implementation Analysis for Flashlight

## Executive Summary

This analysis examines how five production ML frameworks implement VJP (vector-Jacobian product) gradient transforms, organized by design decision. Each section compares framework approaches and recommends a path for Flashlight's C++ JIT IR.

The core finding: **MLX's design is the closest match for Flashlight**, with one key difference. MLX's VJP rules are virtual methods on Primitive objects that take `(primals, cotangents, argnums, outputs)` and return cotangent arrays. The backward pass is a reverse topo-sort walk over the traced forward graph, calling each primitive's `vjp()` method and accumulating cotangents in a hash map. This is almost exactly what Flashlight's TODO.md already sketches, and is the right approach. The specific design decisions below refine the details.

---

## Decision 1: VJP Rule Signature

### How each framework does it

**JAX** decomposes VJP into two parts: JVP rules (forward-mode) and transpose rules (reverse-mode). The signature is indirect:
- JVP rule: `(primals, tangents, **params) -> (primal_out, tangent_out)`
- Transpose rule: `(cotangent, *args_with_undefined_primals, **params) -> cotangent_inputs`

JAX marks non-differentiated arguments as `UndefinedPrimal` sentinels. The transpose rule sees which arguments are "undefined" (being differentiated) and returns cotangents only for those. This decomposition is elegant for composability but adds conceptual overhead.

Registration uses global dictionaries:
```python
primitive_jvps: dict[Primitive, Callable] = {}
primitive_transposes: dict[Primitive, Callable] = {}
```
with helper functions `deflinear(prim, transpose_rule)` for linear ops and `defbilinear(prim, lhs_rule, rhs_rule)` for bilinear ops (like matmul, multiply).

**MLX** uses a single virtual method on the Primitive class:
```cpp
virtual std::vector<array> vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs);
```
- `primals`: the original inputs to the forward op
- `cotangents`: one per output (incoming gradient)
- `argnums`: which input gradients are actually needed (optimization)
- `outputs`: the forward op's outputs (for ops like exp whose backward needs the output)

Returns: one cotangent array per argnum entry.

Concrete examples from MLX source:
```cpp
// Add::vjp -- linear op, cotangent passes through
std::vector<array> Add::vjp(...) {
  if (argnums.size() == 1) return cotangents;
  return {cotangents[0], cotangents[0]};
}

// Multiply::vjp -- bilinear: d(a*b)/da = b*ct, d(a*b)/db = a*ct
std::vector<array> Multiply::vjp(...) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(conjugate(primals[1 - arg], stream()), cotangents[0], stream()));
  }
  return vjps;
}

// Matmul::vjp -- dL/dA = ct @ B^T, dL/dB = A^T @ ct
std::vector<array> Matmul::vjp(...) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) vjps.push_back(matmul(cotan, complex_transpose(primals[1]), s));
    else          vjps.push_back(matmul(complex_transpose(primals[0]), cotan, s));
  }
  return vjps;
}

// Exp::vjp -- needs the OUTPUT, not the input: d(exp(x)) = exp(x)*ct
std::vector<array> Exp::vjp(...) {
  return {multiply(cotangents[0], outputs[0], stream())};
}

// Log::vjp -- delegates to jvp (self-adjoint): d(log(x)) = ct/x
std::vector<array> Log::vjp(...) {
  return jvp(primals, cotangents, argnums); // same as JVP for this op
}
```

**PyTorch** uses `torch.autograd.Function` with `forward()`/`backward()` static methods and a context object:
```python
class Exp(Function):
    @staticmethod
    def forward(x): return x.exp()
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
```
The context is mutable state that bridges forward and backward -- the antithesis of functional.

**tinygrad** uses pattern matching on UOp types:
```python
# Addition: gradient passes through to both inputs
# Multiply: d(a*b)/da = b*ct, d(a*b)/db = a*ct
# Exp: d(exp(x)) = exp(x)*ct  (uses ret, the forward output)
```
Registered via a `PatternMatcher` that rewrites UOp graphs.

### Recommendation for Flashlight

Use the **MLX signature**, adapted for Flashlight's Node-based IR. The rule function should be:

```cpp
using VJPRule = std::function<std::vector<NodePtr>(
    const Node& fwd,           // the forward node
    const std::vector<NodePtr>& cotangents,  // per-output cotangents
    const std::vector<int>& argnums          // which input grads needed
)>;
```

Key difference from the TODO.md sketch: **multi-output cotangents**. The TODO.md uses a single `NodePtr ct`, but multi-output nodes (like BatchNorm producing `{y, mean, var}`) need per-output cotangents. MLX handles this correctly. The `argnums` parameter is an important optimization -- it avoids computing unused gradients (e.g., if only the weight gradient is needed for a matmul, skip the input gradient).

The forward node's outputs are accessible from the node itself (via `getResult()` or by referencing the forward node as an input to the gradient expression). This means we do NOT need a separate `outputs` parameter as MLX has -- in Flashlight's JIT IR, the forward node IS the output.

Alternatively (and this is the slightly cleaner version matching MLX more closely), since VJP rules are written as tensor operations (not node-level manipulation), the higher-level API is:

```cpp
using VJPRule = std::function<std::vector<Tensor>(
    const std::vector<Tensor>& primals,     // forward inputs as lazy tensors
    const std::vector<Tensor>& cotangents,  // per-output cotangents as lazy tensors
    const std::vector<int>& argnums,        // which input grads needed
    const std::vector<Tensor>& outputs      // forward outputs (for exp, sigmoid, etc.)
)>;
```

This is the "VJP rules as tensor code" approach from TODO.md line 1080. The tensor operations inside the rule build JIT nodes lazily, so the rule simultaneously describes the math and constructs the backward graph. This is more readable and more composable than node-level manipulation.

---

## Decision 2: Where Residuals Live

"Residuals" are values from the forward pass that the backward pass needs. For example, `sigmoid_backward(ct) = ct * sig * (1 - sig)` needs `sig` (the sigmoid output).

### How each framework does it

**JAX** has the most sophisticated approach. During linearization (VJP's forward pass), JAX tracks which values are needed by the backward pass as "residuals." These are explicitly separated into:
- Values that are already inputs to the function (forwarded by reference, `in_fwd`)
- Values that are already outputs of the function (forwarded by reference, `out_fwd`)
- "True" residuals that must be saved (neither input nor output)

This optimization avoids duplicating tensors that are already available. The residuals are captured as constants in a `ClosedJaxpr` and threaded from the forward to the backward jaxpr.

```python
residual_avals, nzs_out, jaxpr, env, in_fwd, out_fwd = linearize_outs_thunk()
residuals = subs_list2(in_fwd, out_fwd, primals, primals_out, non_fwd_res)
```

For custom VJPs, the user explicitly returns residuals:
```python
def f_fwd(x, y):
    return f(x, y), (jnp.cos(x), jnp.sin(x), y)  # (output, residuals)
def f_bwd(res, g):
    cos_x, sin_x, y = res
    return (cos_x * g * y, sin_x * g)
```

**MLX** takes a much simpler approach: **residuals live in the forward graph**. The VJP rule receives `primals` (forward inputs) and `outputs` (forward outputs) directly. If the backward needs `exp(x)`, it references the forward Exp node's output. The forward graph is retained during backward via a `RetainGraph` RAII guard:

```cpp
std::vector<array> vjps;
{
    detail::RetainGraph retain;  // prevents forward graph cleanup
    vjps = a.primitive().vjp(a.inputs(), cotangents, argnums, outputs);
}
```

This is simple and correct: the forward graph already contains everything the backward needs. No separate residual mechanism.

**PyTorch** uses explicit `ctx.save_for_backward(*tensors)`. Tensors saved this way are stored on the autograd node and reference-counted. This is the most error-prone approach (forget to save, memory leaks from over-saving, saved tensor hooks complexity).

**tinygrad** has no explicit residuals. The gradient rules reference forward UOps directly from the UOp graph (e.g., `ret` for the forward output, `src[0]` and `src[1]` for inputs).

### Recommendation for Flashlight

**Residuals live in the forward JIT graph.** This is what the TODO.md already describes and is the right choice. The VJP rule receives the forward node (which references its inputs and, after evaluation, its outputs). The rule creates backward nodes that reference forward nodes as inputs.

The key insight: in a lazy evaluation framework, the forward graph is still alive when the backward pass runs because no evaluation has happened yet. The backward nodes reference forward nodes via shared_ptr, keeping them alive. There is no separate "save for backward" mechanism -- the graph IS the saved state.

For forward nodes whose output is needed by backward (exp, sigmoid, softmax):
- The VJP rule references the forward node directly. E.g., for Exp, the backward expression `ct * exp(x)` becomes `ct * forward_output` where `forward_output` points to the Exp node.
- No copies, no separate storage. The shared_ptr reference keeps the forward node alive.

For rematerialization/gradient checkpointing: a later pass can replace "reference forward output" with "recompute forward output from inputs." This is a graph transformation, not a change to the VJP rule.

---

## Decision 3: Backward Pass Algorithm

### How each framework does it

**JAX** uses a sophisticated multi-step process:
1. Trace the function to produce a Jaxpr (the IR)
2. Linearize: produce a forward jaxpr and collect residuals
3. Transpose: walk the linearized jaxpr in reverse, applying transpose rules
4. The `backward_pass3()` function iterates equations in reverse, using `GradAccum` objects for gradient accumulation

The accumulation uses two classes:
```python
class ValAccum(GradAccum):    # standard: accumulate via addition
    def accum(self, x):
        self.val = add_tangents(self.val, x)
class RefAccum(GradAccum):    # for refs: accumulate via ref update
    def accum(self, x):
        self.ref.addupdate(x)
```

**MLX** has the cleanest implementation (complete code shown earlier). Three phases:
1. **Trace**: wrap primals in Copy primitives, run function, mark dependents as tracers
2. **Topo sort**: DFS from outputs to inputs, building a tape of nodes that need gradients. Uses `calc_grad` set to track which nodes are on a gradient-requiring path.
3. **Backward**: iterate tape in reverse, call `primitive.vjp()`, accumulate cotangents in `cotan_map` hash map (keyed by array ID).

The accumulation is simple addition:
```cpp
if (cotan_map.find(in_id) != cotan_map.end()) {
    cotan_it->second = add(cotan_it->second, vjps[i], s);  // accumulate
} else {
    cotan_map.insert({in_id, vjps[i]});  // first cotangent
}
```

**PyTorch** uses the C++ autograd engine which does BFS/DFS over the autograd graph, executing `backward()` functions and accumulating gradients into `.grad` fields.

**tinygrad** topologically sorts the UOp graph, walks in reverse, applies pattern-matched gradient rules, and accumulates via `grads[k] = grads[k] + v`.

### Recommendation for Flashlight

Follow MLX's three-phase algorithm, adapted for the JIT IR:

1. **No tracing needed** -- Flashlight's JIT graph already exists (it is built lazily by tensor operations). The forward graph IS the trace.

2. **Topo sort from loss node**: DFS from the loss node, building a tape of nodes on gradient-requiring paths. A node is on a gradient path if any of its inputs are parameters (marked `requiresGrad`) or transitively depend on parameters.

3. **Reverse walk + accumulation**: Iterate tape in reverse, dispatch to VJP rules, accumulate cotangents in `unordered_map<Node*, NodePtr>`.

The critical difference from MLX: Flashlight does not need the "tracer" mechanism. MLX wraps inputs in Copy primitives to mark them as tracers because MLX's arrays don't inherently track their computation graph -- the trace creates it. Flashlight's tensors already have a JIT graph attached. The backward walk simply traverses the existing graph.

Algorithm sketch:
```
backward(lossNode, params):
  // Phase 1: identify gradient-requiring subgraph
  calc_grad = {p.node() for p in params}
  tape = []
  visited = {}
  for node in DFS(lossNode):
    if any(input in calc_grad for input in node.inputs()):
      tape.push(node)
      calc_grad.insert(node)

  // Phase 2: reverse walk
  cotan_map = {lossNode: ScalarNode(1.0)}
  for node in reverse(tape):
    if node not in cotan_map: continue
    ct = cotan_map[node]
    argnums = [i for i, inp in enumerate(node.inputs()) if inp in calc_grad]
    input_cts = vjp_dispatch(node, ct, argnums)
    for i, argnum in enumerate(argnums):
      inp = node.inputs()[argnum]
      if inp in cotan_map:
        cotan_map[inp] = add(cotan_map[inp], input_cts[i])
      else:
        cotan_map[inp] = input_cts[i]

  // Phase 3: extract parameter gradients
  return {p: cotan_map[p.node()] for p in params}
```

---

## Decision 4: Static vs Dynamic Registration of VJP Rules

### How each framework does it

**JAX**: Dynamic registration via global dictionaries:
```python
primitive_jvps[mul_p] = mul_jvp
primitive_transposes[mul_p] = mul_transpose
```
Uses `deflinear()`, `defbilinear()`, `defjvp()` helper functions. Rules are registered at module import time. New primitives can register rules dynamically.

**MLX**: Static dispatch via virtual methods. Each Primitive subclass overrides `vjp()`:
```cpp
class Add : public UnaryPrimitive {
    std::vector<array> vjp(...) override;  // inline in the class
};
class Matmul : public UnaryPrimitive {
    std::vector<array> vjp(...) override;
};
```
All VJP rules live in `mlx/primitives.cpp`. Adding a new primitive requires creating a new class with a `vjp()` method. No runtime registration, no function pointer table.

**PyTorch**: Dynamic via autograd Function classes. Users subclass `Function` and define `backward()`. For built-in ops, gradients are defined in `derivatives.yaml` and code-generated into C++ functions.

**tinygrad**: Pattern matching -- gradient rules are entries in a PatternMatcher that matches on UOp type:
```python
pm_gradient = PatternMatcher([
    (UPat(Ops.ADD, ...), lambda ...: (ctx, ctx)),
    (UPat(Ops.MUL, ...), lambda ...: (src[1]*ctx, src[0]*ctx)),
    ...
])
```

### Recommendation for Flashlight

Use a **switch-based dispatch on NodeType**, not virtual methods or dynamic registration. The reasons:

1. Flashlight's Node class is not polymorphic for this purpose -- nodes use `NodeType` enum + `impl<T>()` casts, not virtual dispatch.
2. The number of differentiable primitives is small (~30-40 as TODO.md notes).
3. A switch statement is the fastest dispatch mechanism and most debuggable.
4. Custom ops use a dedicated path (CustomNode checks for a user-provided VJP).

```cpp
std::vector<NodePtr> dispatchVJP(const Node& fwd, const std::vector<NodePtr>& cts,
                                  const std::vector<int>& argnums) {
    switch (fwd.type()) {
        case NodeType::Binary:  return vjpBinary(fwd, cts, argnums);
        case NodeType::Unary:   return vjpUnary(fwd, cts, argnums);
        case NodeType::Matmul:  return vjpMatmul(fwd, cts, argnums);
        case NodeType::Conv2d:  return vjpConv2d(fwd, cts, argnums);
        case NodeType::Reduction: return vjpReduction(fwd, cts, argnums);
        case NodeType::Custom:  return vjpCustom(fwd, cts, argnums);
        // ... ~30 more cases
        default: throw std::runtime_error("No VJP rule for " + nodeTypeToString(fwd.type()));
    }
}
```

Within each case, a sub-switch on the specific op (e.g., `BinaryOp::Add`, `UnaryOp::Exp`) handles the actual gradient math.

However, the TODO.md line 1080 insight is critical: **VJP rules should be written as tensor operations**, not node-level manipulation. So the dispatch function converts nodes to lazy tensors, calls tensor-level VJP functions, and extracts the resulting nodes:

```cpp
std::vector<Tensor> vjpMatmul(const std::vector<Tensor>& primals,
                               const std::vector<Tensor>& cts,
                               const std::vector<int>& argnums,
                               const std::vector<Tensor>& outputs) {
    std::vector<Tensor> result;
    for (int arg : argnums) {
        if (arg == 0) result.push_back(fl::matmul(cts[0], fl::transpose(primals[1])));
        else          result.push_back(fl::matmul(fl::transpose(primals[0]), cts[0]));
    }
    return result;
}
```

This is readable math that also builds the backward JIT graph. The dispatch layer is the switch; the rules themselves are tensor code.

---

## Decision 5: Composability -- How grad(grad(f)) Works

### How each framework does it

**JAX**: VJP composability comes from the JVP/transpose decomposition. `grad(f)` produces a function `g` that, when traced, creates a jaxpr containing transpose operations. `grad(g)` traces that jaxpr and applies transpose rules again. Because every primitive used in the backward pass (add, multiply, matmul) also has JVP and transpose rules, the second differentiation "just works."

The key: the backward pass produces the same types of operations (primitives) that the forward pass uses. Since all primitives have differentiation rules, the backward is itself differentiable.

**MLX**: VJP rules return `array` objects, which are lazy. These arrays have their own primitives (Add, Multiply, Matmul). Calling `vjp()` on the backward graph traverses these new arrays and calls THEIR `vjp()` methods. Higher-order gradients compose because the backward graph is made of the same primitives as the forward graph.

**PyTorch**: Higher-order gradients require `create_graph=True` in `backward()`. This causes backward operations to be recorded on a new autograd graph, enabling a second `backward()` call. The mechanism is different (nested tapes) but achieves the same result.

**tinygrad**: Second-order gradients work because gradient UOps are themselves UOps with gradient rules. `compute_gradient()` can be called on the output of `compute_gradient()`.

### Recommendation for Flashlight

**Composability is automatic** when VJP rules are written as tensor operations. The backward pass creates new JIT nodes (Matmul, Add, Multiply, etc.) in the same graph. These nodes are themselves differentiable because they have VJP rules. `grad(grad(f))` walks the backward graph and applies VJP rules to the backward nodes, producing second-order gradient nodes.

No special mechanism needed. The only requirement: every node type used in backward VJP rules must itself have a VJP rule. Since backward rules use only basic primitives (matmul, add, multiply, transpose, reshape, reduce_sum), and these all have VJP rules, composability is guaranteed.

Example: `grad(grad(x -> x^2))`:
- Forward: `x^2` = `mul(x, x)`
- First grad: `vjp(mul) = [ct * x, ct * x]` summed = `2 * ct * x` = `2x` (with ct=1)
- Second grad: this expression contains `mul(2, x)`, so `vjp(mul(2, x)) = [ct * x, ct * 2]`. The gradient w.r.t. x is `2 * ct = 2`. Correct (second derivative of x^2 is 2).

---

## Decision 6: Memory Model for Gradient Tensors

### How each framework does it

**JAX**: XLA manages memory. Residuals are allocated during forward, freed after backward uses them. XLA's buffer donation and memory planning optimize this automatically.

**MLX**: The `RetainGraph` RAII guard prevents forward graph cleanup during backward. After backward, forward nodes can be freed. Cotangents are accumulated as new lazy arrays in the cotan_map. The cotan_map is a local variable -- cotangents are freed when the map goes out of scope (after gradient extraction).

MLX uses `cotan_map.extract()` to move cotangents out of the map when consumed, reducing peak memory.

**PyTorch**: Saved tensors are reference-counted. After backward, autograd nodes are freed (unless `retain_graph=True`), releasing saved tensors. The gradient accumulation buffer (`.grad`) persists on the parameter tensor.

**tinygrad**: Gradients are UOps in the graph. Memory is managed by the UOp scheduler.

### Recommendation for Flashlight

Gradient tensors are **lazy JIT nodes** that participate in the JIT graph like any other computation. This means:

1. **No separate memory management**: Gradient nodes are JIT nodes. They are evaluated by the evaluator and use the same buffer management as forward nodes.

2. **Forward-backward graph coexistence**: The backward nodes reference forward nodes via `shared_ptr<Node>`. Forward nodes are kept alive as long as backward nodes reference them. After evaluation (and if no external references remain), nodes are freed.

3. **Evaluation trigger**: `param.grad()` returns a lazy Tensor (backed by gradient JIT nodes). The gradient is only computed when the user accesses it (prints, reads scalar, passes to optimizer). This defers evaluation until necessary.

4. **Optimization opportunity**: Since both forward and backward are in the same JIT graph, the evaluator can schedule them together. A training step produces one connected DAG (forward + loss + backward + optimizer update). The evaluator can fuse across the forward-backward boundary, schedule communication overlap, and optimize memory allocation across the entire step.

---

## Decision 7: Custom VJP for User-Defined Operations

### How each framework does it

**JAX** has the most developed custom VJP API:
```python
@jax.custom_vjp
def f(x, y):
    return jnp.sin(x) * y

def f_fwd(x, y):
    return f(x, y), (jnp.cos(x), jnp.sin(x), y)  # (output, residuals)

def f_bwd(res, g):
    cos_x, sin_x, y = res
    return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)
```

The fwd function returns `(output, residuals)`. The bwd function takes `(residuals, cotangent)` and returns input cotangents. Residuals can be arbitrary pytrees.

**MLX** uses `CustomTransforms` primitive:
```cpp
std::vector<array> CustomTransforms::vjp(...) {
    std::vector<array> inputs(primals.begin(), primals.end() - num_outputs_);
    auto all_vjps = vjp_fun_(inputs, cotangents, outputs);
    // select vjps for argnums
}
```
The user provides a VJP function when creating the custom op.

**PyTorch** uses `torch.autograd.Function`:
```python
class MyOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x
```

### Recommendation for Flashlight

Two mechanisms, matching TODO.md:

**Option A: Explicit VJP on CustomNode** (for opaque kernels):
```cpp
auto result = fl::customOp("my_fused_op", {x, w},
    /*evalFn=*/ [](const std::vector<Tensor>& inputs) -> Tensor { ... },
    /*vjpFn=*/  [](const std::vector<Tensor>& primals,
                   const std::vector<Tensor>& cts,
                   const std::vector<int>& argnums,
                   const std::vector<Tensor>& outputs) -> std::vector<Tensor> { ... }
);
```

**Option B: Composition** (preferred for most cases):
```cpp
Tensor myOp(const Tensor& x, const Tensor& w) {
    return fl::relu(fl::matmul(x, w) + b);  // all primitives have VJP rules
}
// grad(myOp) works automatically via chain rule
```

Option B is the JAX/MLX idiom and should be strongly preferred. Option A is the escape hatch.

---

## Decision 8: Handling Broadcasting in Backward

A subtle but important detail: when the forward operation broadcasts (e.g., adding a bias `[N,C,H,W] + [1,C,1,1]`), the backward must reduce the cotangent to the smaller shape.

### How each framework does it

**MLX**: VJP rules return cotangents with the output shape. Broadcasting reduction is handled implicitly -- MLX's `add()` handles shape mismatches.

**Flashlight's current autograd** (`Functions.cpp`): Uses `fl::detail::sumAs(gradOutput, originalShape)` to reduce gradients to the input shape. This is the correct approach and must be preserved in VJP rules.

### Recommendation for Flashlight

VJP rules for broadcasting ops (add, sub, mul, div) must include `sumAs` reduction:
```cpp
// Add VJP with broadcast handling
case BinaryOp::Add: {
    std::vector<Tensor> result;
    for (int arg : argnums) {
        result.push_back(fl::detail::sumAs(cts[0], primals[arg].shape()));
    }
    return result;
}
```

This matches the existing `autogradAdd` pattern but expressed as a VJP rule.

---

## Decision 9: Enzyme-Style LLVM-Level AD vs. IR-Level VJP

### How Enzyme works

Enzyme differentiates at the LLVM IR level using an instruction visitor pattern:
1. **Activity analysis**: determines which LLVM values affect the output (analogous to determining which nodes need gradients)
2. **Shadow memory**: creates "shadow" copies of pointers for storing adjoints
3. **Adjoint generation**: for each LLVM instruction, generates the corresponding backward instruction using the chain rule
4. **Modes**: ReverseModePrimal (augmented forward), ReverseModeGradient (backward), ReverseModeCombined

The core loop is: for each instruction, dispatch on instruction type (load, store, binary op, call, etc.), generate adjoint code using `addToDiffe()` for accumulation and `applyChainRule()` for chain rule application.

### Is this applicable to Flashlight?

**Partially.** The algorithmic principle is identical: reverse walk, per-instruction adjoint rules, accumulate via addition. But the level of abstraction differs hugely:
- Enzyme works on scalar LLVM instructions (add, mul, load, store)
- Flashlight works on tensor-level operations (matmul, conv2d, reduce_sum)

Enzyme's activity analysis concept maps directly to Flashlight's `calc_grad` set (determining which nodes need gradients). The shadow memory concept is unnecessary -- Flashlight uses a hash map for cotangent accumulation.

**The one useful insight from Enzyme**: it handles the "augmented forward" pattern well. Some backward computations need values from the forward pass that aren't the input or output (intermediate values). Enzyme explicitly augments the forward pass to return these. For Flashlight, this maps to: if a VJP rule needs a forward intermediate (e.g., the pre-softmax logits in softmax backward), the rule should express this as a recomputation from inputs rather than requiring special save mechanisms.

---

## Appendix A: Framework Comparison Table

| Aspect | JAX | MLX | PyTorch | tinygrad | Flashlight (proposed) |
|--------|-----|-----|---------|----------|----------------------|
| Rule signature | JVP + transpose | `vjp(primals, cts, argnums, outputs)` | `backward(ctx, grad)` | pattern match | `vjp(primals, cts, argnums, outputs)` |
| Registration | global dict | virtual method | Function subclass | PatternMatcher | switch on NodeType |
| Residuals | explicit, optimized | forward graph reference | ctx.save_for_backward | implicit in UOp graph | forward graph reference |
| Backward walk | reversed jaxpr eqns | reversed tape | C++ engine | reversed topo sort | reversed tape |
| Accumulation | GradAccum objects | hash map + add | .grad field | dict + add | hash map + add |
| Composability | via JVP/transpose | via lazy array vjp | create_graph=True | recursive compute_gradient | via lazy tensor ops |
| Custom ops | @custom_vjp | CustomTransforms | Function subclass | N/A | customOp() + vjpFn |
| Memory | XLA manages | RetainGraph guard | refcount + save | UOp scheduler | JIT graph shared_ptr |
| Code size | ~2000 lines (ad.py) | ~3000 lines (primitives.cpp) | ~5000 lines (engine + functions) | ~200 lines (gradient.py) | est. ~1500 lines |

## Appendix B: Complete VJP Rule List for Flashlight

Based on Flashlight's NodeType enum, the required VJP rules:

| NodeType | Differentiable? | VJP complexity | Notes |
|----------|----------------|----------------|-------|
| Binary (Add) | Yes | Trivial | ct passes through, sumAs for broadcast |
| Binary (Sub) | Yes | Trivial | ct, -ct |
| Binary (Mul) | Yes | Simple | ct*other_input |
| Binary (Div) | Yes | Medium | ct/rhs, -ct*lhs/rhs^2 |
| Binary (Pow) | Yes | Medium | exponent and base gradients |
| Unary (Exp) | Yes | Trivial | ct * output |
| Unary (Log) | Yes | Trivial | ct / input |
| Unary (Sin) | Yes | Simple | ct * cos(input) |
| Unary (Cos) | Yes | Simple | -ct * sin(input) |
| Unary (Tanh) | Yes | Simple | ct * (1 - output^2) |
| Unary (Sigmoid) | Yes | Simple | ct * output * (1 - output) |
| Unary (Negative) | Yes | Trivial | -ct |
| Unary (Abs) | Yes | Simple | ct * sign(input) |
| Unary (Sqrt) | Yes | Simple | ct / (2 * output) |
| Unary (Erf) | Yes | Medium | ct * 2/sqrt(pi) * exp(-x^2) |
| Cast | Yes | Trivial | cast ct back to input dtype |
| Matmul | Yes | Medium | ct @ B^T, A^T @ ct |
| Conv2d | Yes | Hard | transposed conv for input grad, conv for weight grad |
| Reduction (Sum) | Yes | Simple | broadcast ct to input shape |
| Reduction (Mean) | Yes | Simple | broadcast ct / num_elements |
| Reduction (Max) | Yes | Medium | ct * (input == max_value) mask |
| Reshape | Yes | Trivial | reshape ct back to input shape |
| Transpose | Yes | Trivial | inverse transpose of ct |
| Index | Yes | Medium | scatter ct into zeros at indices |
| IndexedUpdate | Yes | Medium | index for updated positions, identity elsewhere |
| Scalar | No | N/A | constants have no gradient |
| Value | No | N/A | leaf values |
| Custom | Maybe | Varies | user-provided VJP or error |
| Projection | Yes | Trivial | identity on projected output |
| BatchNorm | Yes | Hard | multi-output, complex gradient |
| Pool2d | Yes | Hard | unpooling / index-based scatter |
| Transfer | Yes | Trivial | transfer ct back to source device |
| MultiValue | Yes | Medium | route ct to corresponding sub-outputs |
| FusedElementwise | Yes | Medium | decompose into constituent ops for VJP |
| FusedReduceElementwise | Yes | Hard | decompose into reduce + elementwise |

Estimated: ~35 VJP rule implementations (some grouped by NodeType with sub-switches).

## Appendix C: The Two-Level API

**Level 1: PyTorch-compatible (for most users)**
```cpp
auto loss = model.forward(input);
loss.backward();
for (auto& [name, param] : model.params()) {
    optimizer.step(param, param.grad());
}
```
`backward()` internally walks the JIT graph, applies VJP rules, stores gradient nodes on parameters. `.grad()` returns a lazy Tensor.

**Level 2: Functional (for composition, power users)**
```cpp
auto lossFunc = [&](const std::vector<Tensor>& params) -> Tensor {
    return model.forward(input, params);  // parameterized
};
auto [loss, grads] = fl::valueAndGrad(lossFunc)(params);
// or:
auto hessVecProduct = fl::grad(fl::grad(lossFunc));
```

Both APIs use the same underlying VJP machinery. Level 1 is sugar over Level 2.
