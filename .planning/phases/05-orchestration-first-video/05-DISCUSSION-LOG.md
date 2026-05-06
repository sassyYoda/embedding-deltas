# Phase 5: Orchestration & First-Video End-to-End - Discussion Log

> **Audit trail only.** Decisions in `05-CONTEXT.md`.

**Date:** 2026-05-06
**Phase:** 5 — Orchestration & First-Video End-to-End
**Mode:** `--auto`

| Area | Decision |
|---|---|
| CLI surface | Spec §9 verbatim — no extra flags |
| `run()` signature | Keyword-only after `video_path` |
| Stage banners | Match spec §9 verbatim `[1/6]…[6/6]` |
| Determinism env vars (Pitfall 14) | Module-top BEFORE imports |
| MPS deterministic mode | `warn_only=True` (raises strict on MPS) |
| JSON schema | Spec §8 verbatim, field order preserved |
| `mad_score` | `scores[peak_idx]` (Phase 2 output) |
| `raw_cosine_delta` | `raw_deltas[peak_idx]` (Phase 5 keeps reference to raw deltas in `run()`) |
| `peak_idx` recovery | `{peak_time: peak_idx}` dict built in `run()` between select_peaks and build_clips |
| `coincides_with_pelt_changepoint` (Pitfall 17) | Strict three-state: `null`/`true`/`false`, never omitted |
| Float rounding (Pitfall 16) | 3dp seconds, 4dp scores, at JSON-assembly time |
| Tuning video | JT (Justin Timberlake) |
| Frozen parameters | argparse defaults `(1.5, 15.0, 3.0)` (revisit if JT reel quality is poor) |
| Two-run reproducibility | Sanity test in `__main__`; differences documented as non-blocker |
| Visual qualitative gate | `__main__` prints reel + video paths; user manually plays + reports verdict |

## Deferred

- `--diagnostics` flag → v2
- `--cache` flag → v2
- Logging library → out of scope (use `print`)
- Crash recovery → out of scope
- JSON schema validation → out of scope
