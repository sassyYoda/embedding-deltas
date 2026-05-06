# Phase 2: Signal Processing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `02-CONTEXT.md`.

**Date:** 2026-05-06
**Phase:** 2 — Signal Processing
**Mode:** `--auto` (recommended option auto-selected; no interactive AskUserQuestion calls)
**Areas discussed:** Cosine deltas, alignment helper, median filter source, MAD floor, PELT lazy import, verification harness, fixture output, multi-video awareness

---

## Area A — Cosine Delta Computation Path

| Option | Description | Selected |
|--------|-------------|----------|
| Spec §4 verbatim: `np.sum(emb[:-1] * emb[1:], axis=1)` + `np.clip(-1, 1)` + `1 - dot` | Trust Phase 1's L2-norm assertion; don't re-normalize. | ✓ |
| Re-normalize defensively inside compute_deltas | Adds ~50 ms but Phase 1 already guards. | |

**Auto-selected:** Spec verbatim (recommended).
**Notes:** Phase 1 D-13's `np.allclose(norms, 1.0, atol=1e-5)` hard-assert means re-normalizing would be redundant and hide bugs in extract.py.

---

## Area B — Median Filter Source

| Option | Description | Selected |
|--------|-------------|----------|
| `scipy.signal.medfilt(raw, kernel_size=5)` (spec §5 verbatim) | Zero-pads edges → produces phantom dips in first/last 2 samples (Pitfall 9). | |
| `scipy.ndimage.median_filter(raw, size=5, mode='reflect')` (research recommendation) | Reflects boundaries → preserves edge values; behaviorally identical interior; matches "edge-preserving" intent the spec invokes. | ✓ |

**Auto-selected:** `scipy.ndimage.median_filter(mode='reflect')` (research-recommended).
**Notes:** Documented deviation from spec verbatim. The deviation is justified by Pitfall 9 in research/PITFALLS.md and is consistent with the spec's stated rationale for choosing median over Gaussian (edge preservation). Inline `# Deviation from spec §5: ...` comment will explain.

---

## Area C — MAD-Zero Window Floor

| Option | Description | Selected |
|--------|-------------|----------|
| `if mad > 1e-8` (spec §5 verbatim) | Too lenient; static-footage windows produce MAD in `[1e-7, 1e-3]` that round-trip through float math and produce ceiling-pegged scores from divisions by near-zero. | |
| `if mad > 1e-3` (research recommendation) | Defensible static-footage floor; emit `0.0` below this. Pitfall 10. | ✓ |

**Auto-selected:** `1e-3` floor.
**Notes:** Documented deviation. The §0.5 print includes percentage of zero-MAD samples to surface if the threshold trips too often (spec recommends `<5%` typical).

---

## Area D — Alignment Invariant Helper

| Option | Description | Selected |
|--------|-------------|----------|
| Single helper `score_index_to_timestamp(idx, ts) -> ts[idx+1]` in `signal_processing.py` | One conversion site; Phase 3 imports it; locks invariant in code. | ✓ |
| Inline `ts[idx+1]` everywhere it's needed | More flexible but invites drift between Phase 3 sites. | |

**Auto-selected:** Single helper.
**Notes:** Matches research/SUMMARY.md cross-cutting invariant: "signal_processing.py is index-space only; clip_selection.py is the single conversion seam." The helper makes that policy enforceable.

---

## Area E — Synthetic Alignment Test Fixture

| Option | Description | Selected |
|--------|-------------|----------|
| Generate at runtime in `__main__` (in-memory) | Small fixture (~K × 768 × 4 bytes for K~50); no on-disk persistence needed. | ✓ |
| Save to `output/cache/synthetic_*.npy` as a reference fixture | Allows external test reuse but bloats cache dir. | |

**Auto-selected:** Runtime generation.
**Notes:** Test runs in `__main__` once per Phase 2 verification. No persistent value to caching.

---

## Area F — PELT Lazy-Import Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Lazy-import inside `detect_changepoints()` body | When `--pelt` is off, `ruptures` is never loaded. Verifiable via `'ruptures' not in sys.modules`. | ✓ |
| Module-level `import ruptures` with try/except | Loads `ruptures` whenever `signal_processing` is imported. | |

**Auto-selected:** Inside-function lazy import.
**Notes:** Matches research/PITFALLS.md and CONTEXT 01's cross-cutting `--pelt` orthogonality. Verified at end of non-PELT runs.

---

## Area G — Verification Harness Style

| Option | Description | Selected |
|--------|-------------|----------|
| `if __name__ == "__main__":` block in `signal_processing.py` (matches Phase 1 D-16) | Consistent with extract.py. Runnable as `python signal_processing.py`. | ✓ |
| Separate `tests/test_signal_processing.py` | Adds pytest scaffolding; not needed for prototype. | |

**Auto-selected:** Inline `__main__` block.
**Notes:** D-16 from Phase 1 set the precedent. Same pattern.

---

## Area H — Fixture Output

| Option | Description | Selected |
|--------|-------------|----------|
| Default-on `--save-fixture` writes `scores.npy` (and `changepoints.npy` when `--pelt`) | Saves Phase 3 from re-running Phase 2 every dev iteration. Internal dev tool. | ✓ |
| No fixture output; Phase 3 always re-runs Phase 2 | Adds ~5 s per Phase 3 iteration; trivial cost but fixtures are basically free. | |

**Auto-selected:** Save fixture.
**Notes:** Matches Phase 1 D-18 pattern. Cache stays gitignored under `output/`.

---

## Claude's Discretion

- Internal docstring style, type-hint style.
- Synthetic-test K value (default ~50).
- Whether `mad_normalize` is a Python loop (matches spec §5 snippet) or vectorized (defer; loop is fine on 2,300 samples).
- Tolerance constants: `ε=1e-9` for synthetic alignment test.

## Deferred Ideas

- Vectorized rolling-MAD via stride_tricks → v2 perf
- Multi-video signal comparison → Phase 5
- Configurable kernel/window sizes → locked by spec §5
- PELT with `cosine` model instead of `rbf` → v2
- Save smoothed-deltas as separate fixture → defer unless Phase 3 needs it
