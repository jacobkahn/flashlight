# History of the AI-assisted modernization sessions

The modernization was done by the maintainer working with AI coding assistants (Claude
Code from 2026-02, Codex in 2026-05/06, Claude Code again from 2026-06). This directory
preserves what those sessions decided and why, so the work can continue anywhere.

| File | Contents | Source |
|---|---|---|
| [MAINTAINER-DIRECTIVES.md](MAINTAINER-DIRECTIVES.md) | **Start here.** The maintainer's standing design principles and working preferences, by theme | All sessions |
| [sessions-2026-02-to-06-claude-code.md](sessions-2026-02-to-06-claude-code.md) | Digest of ~45 sessions (setup, laziness, multi-output IR, fusion, async materialization, transfers, API cleanup, passes) | Prompts only (transcripts not retained) + `TODO.md` |
| [sessions-2026-05-codex.md](sessions-2026-05-codex.md) | First-class IR node promotions and their rewrites; MoE / NVRTC / kernel-DSL design memo; capability-query answer | Full transcripts |
| [sessions-2026-10-to-11-claude-code.md](sessions-2026-10-to-11-claude-code.md) | The 2026-10-26 assessment and decision framing; pointers into `LOG.md` | Full transcript |
| [prompts-log.md](prompts-log.md) | Every maintainer prompt, chronologically, lightly cleaned | Prompt histories of both tools |

**Provenance and cleaning.** Raw transcripts are not included: they contain tool output,
local paths and machine details. Prompts were cleaned only to remove slash commands,
duplicates, local paths, an internal network tool, and a pasted third-party discussion.
Where only prompts survive, the digests infer outcomes from the prompts and from what is
in `TODO.md` and the code; treat them as summaries, not records.

Pre-2026 history (the original Flashlight) is in the upstream repository and the paper
(arXiv:2201.12465).
