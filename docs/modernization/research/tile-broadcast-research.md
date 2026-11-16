# TileNode Design Analysis: How Production ML Compiler IRs Handle Tile and Broadcasting

## 1. Is Tile a First-Class IR Node in Major Frameworks?

### XLA/HLO

**No dedicated Tile op.** HLO has exactly one broadcast-related opcode: `kBroadcast`. There is no `kTile` or `kRepeat` opcode in the HLO instruction set.

HLO has two user-facing broadcast operations that both lower to `kBroadcast`:

- **`Broadcast(operand, broadcast_sizes)`**: Adds new dimensions on the left by duplicating data. If operand has shape `{b0,...,bM}` and `broadcast_sizes = {a0,...,aN}`, output shape is `{a0,...,aN, b0,...,bM}`. This inserts leading dimensions only.

- **`BroadcastInDim(operand, out_dim_size, broadcast_dimensions)`**: The general form. Expands size and rank by mapping operand dimensions to target dimensions via `broadcast_dimensions`. Each operand dimension must either be 1 (and gets broadcast) or match the target. This is the canonical broadcast operation in XLA.

**How tiling is achieved in XLA**: Since there is no tile op, `jnp.tile` is decomposed into `broadcast_in_dim` + `reshape`. The decomposition:
1. Interleave tile-factor dimensions with original dimensions: reshape `[A, B]` to `[1, A, 1, B]`
2. Broadcast to `[T0, A, T1, B]` using `broadcast_in_dim`
3. Reshape/collapse to `[T0*A, T1*B]`

This is the standard decomposition used by JAX and is algebraically clean.

**Broadcasting in binary ops**: HLO binary operations require operands to have identical shapes. Broadcasting must be made explicit via `broadcast_in_dim` nodes inserted before the binary op. This is a fundamental design choice -- XLA has NO implicit broadcasting.

```
// XLA HLO representation of a + b where a:[3,4], b:[1,4]
%broadcast = f32[3,4] broadcast_in_dim(f32[1,4] %b), dimensions={0,1}
%add = f32[3,4] add(f32[3,4] %a, f32[3,4] %broadcast)
```

### MLIR StableHLO

**Same as XLA -- no dedicated tile op.** StableHLO mirrors HLO closely.

- **`stablehlo.broadcast_in_dim`**: The primary broadcast operation. Expands dimensions and/or rank by duplicating data. Takes `broadcast_dimensions` attribute mapping operand dims to result dims.

```mlir
%result = "stablehlo.broadcast_in_dim"(%operand) {
  broadcast_dimensions = array<i64: 2, 1>
} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
```

**Broadcasting in binary ops**: Like HLO, StableHLO binary ops require operands to have identical shapes. The constraint is explicit: `type(lhs) = type(rhs) = type(result)`. Broadcasting must be explicit via `broadcast_in_dim`.

**No tile op**: Tiling is decomposed into `broadcast_in_dim` + `reshape` during lowering from higher-level representations.

### TVM Relay/TIR

**Yes, TVM has both tile and broadcast as first-class operations.**

- **`relay.tile(data, reps)`**: Repeats the whole array multiple times. `reps` is a tuple of repetition counts per dimension. Input `[2,3]` with reps `(2,3)` produces `[4,9]`.

- **`relay.broadcast_to(data, shape)`**: Broadcasts tensor to a target shape following NumPy rules. Expands dimensions of size 1. Input `[1,5]` broadcast to `[3,5]` produces `[3,5]`.

- **`relay.broadcast_to_like(data, broadcast_type)`**: Broadcasts to match the shape of another tensor.

**Broadcasting in binary ops**: TVM Relay binary ops handle broadcasting IMPLICITLY. The Relay type system infers output shapes using broadcasting rules. No explicit broadcast node is needed before binary ops.

**At the TIR (low-level) level**: Both tile and broadcast_to lower to loop-based compute definitions. `tile` uses modular indexing: `output[i] = input[i % input_size]`. `broadcast_to` uses conditional indexing: `output[i,j] = input[0 if dim==1 else i, j]`.

### ONNX

**Yes, ONNX has both Tile and Expand as separate ops.**

- **`Tile(input, repeats)`**: Constructs a tensor by replicating input. `repeats` is a 1D int64 tensor specifying repetition counts per dimension. `output_dim[i] = input_dim[i] * repeats[i]`. Explicitly states "no broadcast" -- this is pure data replication.

- **`Expand(input, shape)`**: Broadcasting operation. Follows NumPy broadcasting rules: dimensions are right-aligned, corresponding dimensions must match or one must be 1. Unlike NumPy's `broadcast_to`, Expand allows the target shape to be smaller than input (dimensions that are 1 in the target are kept from the input).

**Key difference**: ONNX keeps these separate because they have fundamentally different semantics:
- Tile: input `[2,3]` with repeats `[2,2]` -> output `[4,6]` (data copied)
- Expand: input `[1,3]` with shape `[2,3]` -> output `[2,3]` (broadcast, may not copy)

**Broadcasting in binary ops**: ONNX binary ops (Add, Mul, etc.) handle broadcasting IMPLICITLY using NumPy broadcasting rules. No explicit Expand node is needed.

### PyTorch FX/Inductor

**PyTorch has three user-facing operations with different semantics:**

- **`torch.Tensor.expand(*sizes)`**: View-based, NO data copy. Uses stride tricks (stride=0 for broadcast dims). `-1` means keep original size. Warning: in-place operations on expanded tensors may give incorrect results due to memory aliasing.

- **`torch.Tensor.repeat(*repeats)`**: Copies data. Repeats tensor along specified dimensions. Note: behaves like numpy.tile, NOT numpy.repeat.

- **`torch.tile(input, dims)`**: Equivalent to `repeat` with dimension alignment (prepends 1s if dims has fewer dimensions than input, unsqueezes if input has fewer dimensions).

**In PyTorch Inductor IR**:
- `expand` lowers to `ExpandView` -- a view node that manipulates strides/sizes without copying data
- `repeat` and `tile` are NOT first-class IR nodes. They decompose to sequences of view operations.
- `broadcast_to` decomposes to `prims.broadcast_in_dim(a, size, dims)` in PyTorch refs

**Broadcasting in binary ops**: PyTorch FX binary ops handle broadcasting IMPLICITLY. Inductor's `broadcast_symbolic_shapes()` resolves shapes at the symbolic level.

**Key insight from PyTorch**: The reference implementation of `expand` decomposes to `prims.broadcast_in_dim(a, shape_, tuple(range(offset, ...)))`, showing that `broadcast_in_dim` is the canonical primitive even in PyTorch.

---

## 2. How Is Broadcasting Represented in Each IR?

| Framework | Broadcasting in Binary Ops | Explicit Broadcast Node | Tile Op |
|-----------|--------------------------|------------------------|---------|
| **XLA/HLO** | Explicit (required) | `broadcast_in_dim` | None (decomposed) |
| **StableHLO** | Explicit (required) | `broadcast_in_dim` | None (decomposed) |
| **TVM Relay** | Implicit | `broadcast_to` (available) | `tile` (first-class) |
| **ONNX** | Implicit | `Expand` (available) | `Tile` (first-class) |
| **PyTorch Inductor** | Implicit | `ExpandView` (IR node) | None (decomposed) |

**Two schools of design emerge:**

**School A -- Explicit Broadcasting (XLA/StableHLO)**:
- Binary ops require same-shaped operands
- `broadcast_in_dim` nodes are inserted before every broadcasting binary op
- Tile decomposes to `broadcast_in_dim` + `reshape`
- Pro: IR is simple, every node's semantics are clear, optimizations are straightforward
- Con: Graph is larger (more nodes), needs canonicalization passes to remove redundant broadcasts

**School B -- Implicit Broadcasting (TVM, ONNX, PyTorch)**:
- Binary ops handle shape mismatch internally
- Separate tile and broadcast ops exist but are used only for explicit user calls
- Pro: Smaller graph, matches user mental model
- Con: Binary op implementations must handle broadcasting, optimization passes must understand implicit broadcasting

---

## 3. Tile vs Expand vs Repeat -- Semantic Differences

### The Fundamental Distinction

These operations form a spectrum from "view" to "copy":

```
expand (view, stride=0)  <-->  broadcast_in_dim (abstract)  <-->  tile/repeat (data copy)
```

#### expand / broadcast_in_dim (No Copy)
- Creates a view with stride=0 on broadcast dimensions
- Memory: O(1) -- no new allocation
- Constraint: can only expand dimensions of size 1
- Example: `[3,1,4]` -> `[3,5,4]` by setting stride[1]=0
- The "input" to expand is semantically a tensor with a size-1 dimension

#### tile / repeat (Data Copy)
- Physically replicates data
- Memory: O(product of tile factors * input size)
- No constraint on input dimension sizes -- any dimension can be tiled
- Example: `[3,4]` tiled by `[2,3]` -> `[6,12]`

#### The Relationship
Broadcasting is a special case of tiling where:
- The input has size 1 on the broadcast dimensions
- The tile factor equals the target size on those dimensions
- The result CAN be represented as a stride-0 view (no copy needed)

Tiling is a generalization of broadcasting:
- `tile([3,4], [2,1])` -> `[6,4]` : the dim-0 has size 3, not 1, so this CANNOT be expressed as broadcast
- `tile([1,4], [3,1])` -> `[3,4]` : this CAN be expressed as broadcast (equivalent to `expand`)

### Canonical Form

Most frameworks treat `broadcast_in_dim` as the canonical form for the "expand" case, and decompose explicit tiling differently:

**JAX/XLA canonical tile decomposition**:
```
tile(x:[A,B], [T0,T1])
  = reshape(broadcast_in_dim(reshape(x, [1,A,1,B]), [T0,A,T1,B], [0,1,2,3]), [T0*A, T1*B])
```

Steps:
1. Reshape to interleave size-1 dims: `[A,B]` -> `[1,A,1,B]`
2. Broadcast the size-1 dims: `[1,A,1,B]` -> `[T0,A,T1,B]`
3. Reshape to collapse: `[T0,A,T1,B]` -> `[T0*A, T1*B]`

**PyTorch canonical tile decomposition**: Similar -- `tile` calls `repeat`, which internally uses `expand` + `reshape`.

---

## 4. How Does Tile Interact with Reshape?

### Algebraic Identities

#### Identity 1: tile(reshape(x)) simplification
```
tile(reshape(x, [1, N]), [M, 1])
  = broadcast_in_dim(reshape(x, [1, N]), [M, N])  -- if tile factor aligns with size-1 dim
```
When a reshape introduces a size-1 dimension and the tile expands exactly that dimension, the pair simplifies to a single `broadcast_in_dim`.

#### Identity 2: reshape(tile(x)) dim adjustment
```
reshape(tile(x:[A], [T]), [T, A])
  = broadcast_in_dim(reshape(x, [1, A]), [T, A])  -- the tile+reshape becomes broadcast
```
When a tile is followed by a reshape that factorizes the tiled dimension back into (tile_factor, original_size), the pair becomes broadcast.

#### Identity 3: Tile with factor 1 is identity
```
tile(x, [1, 1, ..., 1]) = x  -- trivially optimizable
```

#### Identity 4: Consecutive tiles combine
```
tile(tile(x, [A0, A1]), [B0, B1]) = tile(x, [A0*B0, A1*B1])
```

#### Identity 5: Tile + Reduction cancellation
```
sum(tile(x:[1,N], [M,1]), axis=0) = M * x  -- tile then reduce = scale
sum(tile(x:[A,B], [M,1]), axis=0, keepdims=true) = M * sum(x, axis=0, keepdims=true)
```
More generally, when a reduction axis aligns with a tiled dimension:
- If the original dim size is 1: the result is `tile_factor * x` (scalar multiplication)
- If the original dim size > 1: the result is `tile_factor * sum(x, axis, keepdims=true)` then reshape

#### Identity 6: tile + elementwise commute (when shapes allow)
```
f(tile(x, T)) = tile(f(x), T)  -- for any elementwise unary f
tile(x, T) op tile(y, T) = tile(x op y, T)  -- for elementwise binary op when both have same tile
```

### Tile-Reshape Interaction in the IR

For an IR like Flashlight's JIT:
- `tile(reshape(x))` cannot generally be fused into a single node, because the reshape changes the logical layout
- However, the pattern `reshape -> tile -> reshape` (which is the canonical tile decomposition) should be recognized as a single logical tile operation
- An optimizer pass can detect `reshape(broadcast_in_dim(reshape(x)))` patterns and convert them to/from a single TileNode

---

## 5. Gradient of Tile

### Mathematical Derivation

If `y = tile(x, T)` where `x` has shape `[D0, D1, ..., Dn]` and `T = [T0, T1, ..., Tn]`:

The VJP (backward) is: `dx = sumAs(dy, x.shape)`

Where `sumAs` sums over the dimensions that were tiled to collapse them back. Specifically:

```
dy has shape [D0*T0, D1*T1, ..., Dn*Tn]

dx[i0, i1, ..., in] = sum over all (j0, j1, ..., jn) where jk % Dk == ik of dy[j0, j1, ..., jn]
```

This is equivalent to:
1. Reshape `dy` from `[D0*T0, D1*T1, ...]` to `[T0, D0, T1, D1, ...]`
2. Sum over axes `[0, 2, 4, ...]` (the tile-factor axes)
3. Result has shape `[D0, D1, ...]`

### How Frameworks Express This

**Flashlight (current)**:
```cpp
// In autograd/Functions.cpp, tile backward:
Tensor tile(const Tensor& input, const Shape& dims) {
  Tensor result = fl::tile(input, dims);
  Shape idims = input.shape();
  auto gradFunc = [idims](std::vector<Tensor>& inputs, const Tensor& gradOutput) {
    inputs[0].addGrad(Tensor(
        fl::detail::sumAs(gradOutput, idims).astype(inputs[0].type()), false));
  };
  return Tensor(result, {input.withoutData()}, gradFunc);
}
```
Flashlight uses `sumAs(gradOutput, inputShape)` which:
- Right-aligns input dims with gradient dims
- Sums over dimensions where `gradDim != inputDim`
- Reshapes to match input shape

**This is exactly the same pattern used for broadcast gradients:**
```cpp
// tileAs backward:
fl::detail::sumAs(gradOutput, inDims)

// sumAs backward:
fl::detail::tileAs(gradOutput, idims)

// Binary op (add) broadcast gradient:
fl::detail::sumAs(gradOutput, lhsShape)  // for the smaller operand
```

**PyTorch**:
- `expand` backward: `at::sum_to(grad, self.sym_sizes())` -- sum to original shape
- `repeat` backward: `repeat_backward(grad, repeats, self.sym_sizes())` -- a specialized kernel

**XLA/JAX**:
- `broadcast_in_dim` transpose: reduce_sum over the broadcast dimensions. This is the fundamental relationship.
- Since tile decomposes to `broadcast_in_dim` + `reshape`, the gradient naturally decomposes to `reshape` + `reduce_sum`.

### The Duality: tile <-> sumAs

This is a fundamental mathematical duality that Flashlight already captures:

```
forward: tileAs(x, targetShape)    backward: sumAs(grad, x.shape)
forward: sumAs(x, targetShape)     backward: tileAs(grad, x.shape)
```

The gradient of tile is sum. The gradient of sum is tile (broadcast). They are adjoint operations.

---

## 6. Design Recommendation for Flashlight

### Current State

Currently in Flashlight:
- `tile` is a **CustomNode** with name "tile" -- an opaque lambda wrapping `dispatch.tile()`
- Broadcasting in `BinaryNode` is implicit -- `BinaryNode::create` calls `broadcastShapes()` to compute output shape, and the backend binary op handles the actual broadcasting
- The CustomNode for tile is opaque to all optimization passes (elementwise fusion, cuDNN subgraph lowering, etc.)
- `tileAs`/`sumAs` are utility functions, not IR nodes

### Recommendation: Option (c) -- First-class TileNode + Implicit Broadcasting in Binary Ops

**Design**: Add a `TileNode` as a first-class IR node. Keep broadcasting implicit in BinaryNode. Do NOT add a separate BroadcastNode.

**Rationale**:

1. **TileNode is needed because CustomNode is opaque.** The current CustomNode("tile") cannot be:
   - Pattern-matched by optimization passes
   - Fused with adjacent operations
   - Simplified (e.g., tile(x, [1,1]) -> x)
   - Composed algebraically (consecutive tiles, tile+reduction cancellation)
   
2. **A separate BroadcastNode is NOT needed because**:
   - Flashlight's BinaryNode already handles broadcasting implicitly (via `broadcastShapes`)
   - This matches the TVM/ONNX/PyTorch convention where binary ops broadcast internally
   - Adding explicit broadcast nodes before every broadcasting binary op (XLA-style) would bloat the graph and require a canonicalization pass
   - The XLA approach makes sense for XLA because their entire IR is "explicit" (no implicit semantics anywhere), but Flashlight already has implicit broadcasting in BinaryNode

3. **TileNode captures what BroadcastNode would not**:
   - Broadcasting only handles size-1 -> size-N expansion
   - Tiling handles arbitrary replication: size-M -> size-M*K
   - A TileNode subsumes broadcast-expand as the special case where input dim = 1
   - Any "broadcast as explicit node" scenario can use TileNode with appropriate factors

### TileNode Design

```cpp
class TileNode : public NodeTrait<TileNode> {
  Shape tileDims_;  // tile factors per dimension

  static constexpr unsigned kInputIdx = 0;
  struct PrivateHelper {};

 public:
  static constexpr NodeType nodeType = NodeType::Tile;

  TileNode(
      NodePtr input,
      Shape tileDims,
      const Shape& outputShape,
      dtype outputDtype,
      std::shared_ptr<const Stream> stream,
      PrivateHelper);

  static TileNodePtr create(
      NodePtr input,
      Shape tileDims,
      std::shared_ptr<const Stream> stream = nullptr);

  NodePtr input() const;
  const Shape& tileDims() const;
};
```

Parameters:
- `input`: the tensor to tile
- `tileDims`: per-dimension tile factors. `tileDims[i]` = number of times to replicate along dimension i.
- Output shape: `outputShape[i] = inputShape[i] * tileDims[i]` (computed by `inferTileOutputShape`, which already exists)

### Why NOT option (a) (single node for both tile AND broadcast)?

Option (a) would mean TileNode handles implicit broadcasting inside binary ops. This would require:
- Inserting TileNode before every broadcasting binary op
- Making BinaryNode require same-shaped operands
- This is the XLA approach and would be a massive refactor of Flashlight's existing IR

The cost is not worth it. Flashlight's BinaryNode already handles broadcasting correctly, and the evaluator backends (CPU via OneDNN, CUDA via kernels) all handle broadcasting in their binary op implementations.

### Why NOT option (b) (separate TileNode and BroadcastNode)?

A BroadcastNode would be a TileNode where the input has size-1 dims. There is no semantic distinction that warrants a separate node type:
- Both produce the same kind of output (replicated data)
- Both have the same gradient (sumAs/reduce_sum)
- Both interact with optimization passes the same way
- The only difference is that broadcast can be implemented as a stride-0 view, but this is an implementation detail, not an IR distinction

### Optimization Opportunities with First-Class TileNode

1. **Identity elimination**: `TileNode(x, [1,1,...,1])` -> `x`
2. **Consecutive tile fusion**: `TileNode(TileNode(x, A), B)` -> `TileNode(x, A*B)` (element-wise product of tile factors)
3. **Tile-reduction cancellation**: `ReductionNode(TileNode(x, T), axis)` can be simplified when the reduction axis aligns with a tile factor > 1
4. **Elementwise absorption**: `UnaryNode(TileNode(x, T))` -> `TileNode(UnaryNode(x), T)` -- push tile through elementwise ops (compute on smaller tensor)
5. **cuDNN subgraph lowering**: A TileNode before a Conv2d/MatMul could be recognized and handled by cuDNN's broadcast capabilities
6. **FusedElementwise interaction**: Tile before a fused kernel can be absorbed into the kernel's indexing (use modular indexing)

### Evaluator Implementation

The evaluator would dispatch TileNode to `BackendDispatch::tile`:

```cpp
void Evaluator::evalTileNode(TileNodePtr node) {
  std::function<void()> func = [this, node]() {
    const auto& input = node->input()->getResult().value();
    Tensor out = takeOutputBuffer(node);
    dispatch_.tile(dispatch_.ctx, out, input, node->tileDims());
    node->setResult(std::move(out));
  };
  profile(func, node);
}
```

This is identical to the current CustomNode path but now the node is visible to optimization passes.

### Migration Path

1. Add `NodeType::Tile` to the enum
2. Create `TileNode` class following the pattern of `ReshapeNode`
3. In `JitBackend::tile()`, create a `TileNode` instead of a `CustomNode`
4. Add `evalTileNode` to the evaluator
5. Update `GraphVerifier` to validate TileNode
6. Add identity elimination pass (`tileDims == [1,1,...,1]` -> bypass)
7. Future: add tile-specific optimizations (consecutive fusion, tile+reduction, elementwise commutation)
