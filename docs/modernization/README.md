# Flashlight modernization notes

Everything needed to understand, recover and continue the effort to modernize
Flashlight: what was done, why, what is decided, what is open, and how to build, test
and debug the new stack.

| Document | Read it for |
|---|---|
| [CONTINUING.md](CONTINUING.md) | **Start here on a new machine:** reading order, build check, where work stopped, working with an assistant, and how to split the change set into commits later |
| [STATUS.md](STATUS.md) | Where things stand: build defaults, test results, backend parity, known problems, work in flight, next steps, and a check-in checklist |
| [LOG.md](LOG.md) | Chronological log, from the early phases through the 2026-10/11 review, fixes and decisions |
| [DECISIONS.md](DECISIONS.md) | Decision records D-01…D-12 (accepted, reverted, deferred) and open questions P-1…P-9 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the JIT-only tensor stack fits together; invariants and pitfalls |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Building, test lanes, running tests, runtime switches, debugging techniques |
| [plans/](plans/) | Approved work with file-by-file implementation plans |
| [history/](history/README.md) | The AI-assisted sessions: maintainer directives, session digests (Claude Code and Codex), and the full maintainer prompt log |
| [research/](research/README.md) | The 2026-10 review, external research surveys, backend support matrix, earlier design records |
| [tools/](tools/) | `run_tests.py` (parallel runner over every ctest binary) and `parity.py` (AF vs CPU vs CUDA per-test matrix) |
| [`TODO.md`](../../TODO.md) | The itemized roadmap and task list (phases, design documents, fix lists, friction log) |

## How these notes are maintained
- `TODO.md` is the working checklist. It is append/annotate-only: corrections go inline
  as dated notes, and past records and open items are never deleted.
- New decisions get a record in `DECISIONS.md`; notable work gets a dated entry in
  `LOG.md`; `STATUS.md` is rewritten to match reality whenever it changes.
- Changes beyond local bug fixes (public APIs, IR, evaluator, pass infrastructure,
  semantics, test infrastructure) start as a plan in `plans/` and are agreed before
  implementation.
- Research reports are dated snapshots. Later corrections go in `LOG.md`, `TODO.md` or a
  new report, not into the old report's text.
- Paths are repository-relative. Keep machine-specific details (home directories,
  scratch paths, local environments) out of these files.
