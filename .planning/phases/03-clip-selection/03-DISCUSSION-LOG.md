# Phase 3: Clip Selection - Discussion Log

> **Audit trail only.** Decisions captured in `03-CONTEXT.md`.

**Date:** 2026-05-06
**Phase:** 3 — Clip Selection
**Mode:** `--auto`
**Areas discussed:** Clip tuple structure, find_peaks distance unit, PELT boost integration, partial-clip centering, merge peak_time threading, verification harness, fixture output

---

## Area A — Clip Tuple Structure

| Option | Selected |
|--------|----------|
| 4-tuple `(start, end, score, peak_time)` — `peak_time` threads through merge so spec §8 JSON has `start ≤ peak_timestamp_sec ≤ end` guarantee | ✓ |
| 3-tuple `(start, end, score)` — matches spec §6 snippet verbatim but loses peak_time in merge | |

**Auto-selected:** 4-tuple (Pitfall 19 mitigation; cross-cutting alignment invariant).

---

## Area B — `find_peaks` Distance Unit

| Option | Selected |
|--------|----------|
| `distance=int(min_gap_sec * fps)` — samples (correct per scipy docs + Pitfall 20) | ✓ |
| `distance=min_gap_sec` — seconds (incorrect; would treat seconds as samples → wildly under-spaced peaks) | |

**Auto-selected:** Samples. Inline math at call site + post-call assertion `min(np.diff(peaks)) >= int(min_gap_sec * fps)`.

---

## Area C — PELT Boost Integration

| Option | Selected |
|--------|----------|
| Separate `apply_pelt_boost()` function called from `pipeline.run()` only when `--pelt` is set; never auto-invoked from `select_peaks` | ✓ |
| Inline in `select_peaks` with `changepoints=None` default | |

**Auto-selected:** Separate function. Honors `--pelt` orthogonality (cross-cutting) — the boost path is never traversed when `--pelt` is off.

---

## Area D — Partial-Clip Centering (Pitfall 18)

| Option | Selected |
|--------|----------|
| Center partial clip on `peak_time` (`partial_start = peak_time - remainder/2`, clamp to original clip bounds) | ✓ |
| Truncate at `clip.start` (spec §6 snippet) — produces partial clips that may exclude the actual peak | |

**Auto-selected:** Center on peak_time. Documented deviation from spec §6 snippet; justified by Pitfall 18 + spec §6 intent ("highest-confidence moment" should be in the partial clip).

---

## Area E — Merge `peak_time` Threading (Pitfall 19)

| Option | Selected |
|--------|----------|
| Merged clip keeps higher score AND its associated `peak_time` | ✓ |
| Merged clip keeps higher score; `peak_time` is the midpoint of merged range | |

**Auto-selected:** Higher-scoring peak's `peak_time`. Hard assertion `start ≤ peak_time ≤ end` post-merge guards.

---

## Area F — Verification Harness

| Option | Selected |
|--------|----------|
| `if __name__ == "__main__":` block in `clip_selection.py` (matches Phases 1, 2 D-16, D-34) | ✓ |
| Separate test file (pytest scaffolding) | |

**Auto-selected:** Inline `__main__` block.

---

## Area G — Fixture Output

| Option | Selected |
|--------|----------|
| `--save-fixture` writes `output/cache/{video}_final_clips.json` for Phase 4 dev (NOT spec §8 schema, that's Phase 5) | ✓ |
| No fixture; Phase 4 always re-runs Phase 3 | |

**Auto-selected:** JSON fixture. Same pattern as Phases 1–2; Phase 4 dev iteration becomes instant.

---

## Claude's Discretion

- Type-hint style, docstring style.
- Exact JSON formatting (chose `indent=2` for readability).
- Numerical-safety epsilons: `1e-6` for budget total assertion, `0.95` ratio for peak-too-close-to-edge fallback.

## Deferred Ideas

- NamedTuple / dataclass for clips → v2
- Configurable padding bounds → locked by spec §6
- Configurable PELT boost factor → locked by spec §5
- Multi-objective scoring (diversity penalty) → research/FEATURES.md v3+
- Beam search budget enforcement → spec-locked greedy
- Per-clip metadata in fixture (raw_cosine_delta, mad_score, etc.) → Phase 5 owns spec §8 schema
