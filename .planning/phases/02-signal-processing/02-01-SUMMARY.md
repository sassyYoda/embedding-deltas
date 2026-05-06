---
phase: 02-signal-processing
plan: 01
subsystem: signal-processing
tags: [numpy, scipy, cosine-deltas, median-filter, mad-normalization, pelt, ruptures, lazy-import]

# Dependency graph
requires:
  - phase: 01-frame-extraction-embeddings
    provides: "(N, 768) float32 L2-normalized embeddings + (N,) float64 timestamps cached at output/cache/{video}_{embeddings,timestamps}.npy"
provides:
  - "signal_processing.compute_deltas: (N, 768) → (N-1,) float64 cosine deltas in index space"
  - "signal_processing.smooth_deltas: scipy.ndimage.median_filter mode='reflect' (Pitfall 9-safe)"
  - "signal_processing.mad_normalize: rolling MAD with 1e-3 floor + clip [0, 10] (Pitfall 10-safe)"
  - "signal_processing.detect_changepoints: lazy-imported PELT, list[int] return"
  - "signal_processing.score_index_to_timestamp: the single index→seconds conversion site (Pitfall 8-safe)"
  - "Phase 3 dev fixture: output/cache/justin_timberlake_scores.npy (2268,) float64"
affects: [phase-3-clip-selection, phase-5-pipeline-orchestration]

# Tech tracking
tech-stack:
  added: []  # numpy + scipy already pinned in Phase 1; ruptures already in requirements.txt
  patterns:
    - "Lazy import inside function body (first instance in project) for opt-in dependencies"
    - "Index-space-only module (no timestamps inside the body); single helper guards alignment"
    - "§0.5 verification harness in __main__ matching extract.py pattern"

key-files:
  created:
    - "output/cache/justin_timberlake_scores.npy (2268,) float64 — gitignored Phase 3 dev fixture"
    - "output/cache/justin_timberlake_changepoints.npy (86,) int64 — written when --pelt"
  modified:
    - "signal_processing.py — replaced stub with full implementation + §0.5 harness"

key-decisions:
  - "D-22: compute_deltas returns float64 explicitly (numpy 2.x changed np.sum-with-axis dtype default vs older numpy)"
  - "D-27: scipy.ndimage.median_filter(mode='reflect') instead of the spec's scipy.signal medfilt — Pitfall 9 (zero-padding edge dips)"
  - "D-29: MAD floor at 1e-3 (NOT the spec's much-looser float-precision guard) — Pitfall 10 (static-footage MAD ∈ [1e-7, 1e-3] ceiling-pegs scores)"
  - "D-32: ruptures lazy-imported INSIDE detect_changepoints body — module-top import would load it on every signal_processing import"
  - "Pitfall 10 diagnostic refined: count zero-MAD-branch firings, NOT total zero scores (zeros also include negative residuals clipped by [0, 10] — different concern)"

patterns-established:
  - "Lazy-import contract: opt-in dependencies imported inside function bodies; verified via 'X not in sys.modules' assertion"
  - "Spec-deviation documentation: each deviation tagged in docstring with citation to PITFALLS.md and decision ID"
  - "Synthetic alignment test as the answer to spec §4 'natural choice — verify': two-color fixture, raw-deltas argmax, helper round-trip"

requirements-completed: [SIGD-01, SIGD-02, SIGD-03, SIGS-01, SIGS-02, SIGM-01, SIGM-02, SIGM-03, SIGM-04, SIGP-01, SIGP-02, SIGP-03]

# Metrics
duration: ~25min
completed: 2026-05-06
---

# Phase 2 Plan 01: Signal Processing — compute_deltas → smooth → MAD → opt-in PELT Summary

**Replaced the signal_processing.py stub with five pure-numpy/scipy functions and a §0.5 verification harness that converts the Phase 1 (2269, 768) embedding fixture to a (2268,) float64 anomaly-score array in 0.49 s, with timestamp alignment locked behind a single helper, three pitfalls (8, 9, 10) neutralized by runtime asserts, and ruptures lazy-loaded so non-PELT runs never touch it.**

## Performance

- **Duration:** ~25 min (planning lookup + 2 atomic commits + verification)
- **Started:** 2026-05-06 ~16:30 local
- **Completed:** 2026-05-06 ~16:55 local
- **Tasks:** 2 / 2
- **Files modified:** 1 source file (`signal_processing.py`); 2 fixture files written
- **Harness wall-clock on JT (2,269 frames):** 0.49 s real (well under the 5 s plan target; 30 s safety margin not approached)

## Accomplishments

- All five module functions (`compute_deltas`, `smooth_deltas`, `mad_normalize`, `detect_changepoints`, `score_index_to_timestamp`) ship with locked signatures from D-22..D-32.
- §0.5 9-step harness passes end-to-end on JT fixture; produces `Phase 2 §0.5 verification: PASS` final banner.
- Phase 3 dev fixture `output/cache/justin_timberlake_scores.npy` written: `(2268,) float64`, bounds `[0.0000, 10.0000]`, mean `0.7577`.
- Lazy-import contract verified twice: once inside the harness (after non-PELT path, `'ruptures' not in sys.modules`), once independently (after `compute_deltas` + `smooth_deltas` + `mad_normalize` from a fresh import).
- Synthetic two-color alignment test passes within ε=1e-9 (peak at score-idx 6 ↔ timestamp 3.5 s for K=7 transition at 2 fps).
- `--pelt` round-trip works: 86 changepoints emitted on JT smoothed signal; `'ruptures' in sys.modules` only after the call; `output/cache/justin_timberlake_changepoints.npy` written.
- Smart-default fallback (no positional arg) resolves to `justin_timberlake` via most-recent `*_embeddings.npy` mtime.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement five core functions in signal_processing.py** — `5d3f3f4` (feat)
2. **Task 2: §0.5 verification harness in __main__ + write JT scores fixture** — `baf798b` (feat)

## Files Created/Modified

- `signal_processing.py` — replaced 1-line stub with 313-line implementation (5 functions + 9-step `__main__` harness). Module docstring documents the 3 phase-owned pitfalls and the 2 spec deviations.
- `output/cache/justin_timberlake_scores.npy` — `(2268,) float64`, bounds `[0.0, 10.0]`, mean `0.7577`. Gitignored under `output/`.
- `output/cache/justin_timberlake_changepoints.npy` — `(86,) int64`, only written when `--pelt` flag passed. Gitignored.

## Decisions Made

All decisions inherited from `02-CONTEXT.md` D-22..D-37; honored verbatim except for one diagnostic-logic refinement (see Deviations).

### Pitfall coverage (Phase 2 owned: 8, 9, 10)

- **Pitfall 8 (alignment off-by-one):** `score_index_to_timestamp(score_idx, ts) = float(ts[score_idx + 1])` is the only index→seconds site in `signal_processing.py`. Synthetic two-color test in step 5 of harness asserts `peak_idx == K - 1` AND `score_index_to_timestamp(peak_idx, ts) == K * 0.5` within ε=1e-9. Status: PASS on JT harness run.
- **Pitfall 9 (medfilt edge effects):** `smooth_deltas` uses `scipy.ndimage.median_filter(size=5, mode='reflect')` — NOT `scipy.signal.medfilt`. Grep verifies the forbidden literal `scipy.signal.medfilt` and `from scipy.signal import medfilt` are absent. Spike-injection (`smoothed[100] == 0` after +0.5 spike at index 100 in zeros) and edge-preservation (`smoothed[0] == smoothed[-1] == 7.0` for constant input) both PASS in step 3 of harness.
- **Pitfall 10 (MAD-zero):** `mad_normalize` uses `if local_mad > 1e-3` floor (NOT `1e-8`). Grep verifies `1e-3` is present and `1e-8` is absent. On JT, the actual MAD across all 2,268 score positions falls in `[5.4e-3, 5.1e-2]`, so the zero-MAD branch fires 0 times — confirming static-footage poisoning is not occurring on this fixture.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] §0.5 zero-MAD diagnostic was double-counting clipped negatives**

- **Found during:** Task 2 first harness run on JT.
- **Issue:** The plan's harness computed `zero_mad_count = (scores == 0.0).sum()` and asserted `zero_mad_pct < 5.0`, but `mad_normalize` clips its `[-∞, ∞]` raw output to `[0.0, 10.0]` per spec §5 — so all *negative* normalized residuals (smoothed[i] < local_median) become exactly `0.0` after the lower clip. On JT this gave 1133/2268 = 49.96% scores at zero, even though the actual `local_mad <= 1e-3` zero-MAD branch fires 0 times. Pitfall 10's concern is the static-footage branch firing rate, not the total zero count.
- **Fix:** In step 4 of the harness, count zero-MAD-branch firings explicitly by recomputing `median_abs_deviation(local, scale=1.0) <= 1e-3` for each window position, yielding `zero_mad_branch` (the true diagnostic per Pitfall 10). On JT this is 0/2268 = 0.00%, well below the 5% gate.
- **Files modified:** `signal_processing.py` step 4 of `__main__`.
- **Verification:** Harness passes the `zero_mad_pct < 5.0` assertion with the corrected metric; documented inline with comment citing Pitfall 10.
- **Committed in:** `baf798b` (Task 2 atomic commit).

**2. [Rule 1 — Bug] compute_deltas dtype was float32 (not float64) under numpy 2.x**

- **Found during:** Task 1 acceptance-test Test D.
- **Issue:** Plan D-22 said "Output dtype `float64` (numpy default for `np.sum` on float32 input is float64)". This was true for older numpy; numpy 2.x preserves the input float32 dtype on axis-reductions unless `dtype=` is passed explicitly. `np.sum(emb[:-1] * emb[1:], axis=1)` returned float32, so the contract Phase 3 / Phase 5 expect was violated.
- **Fix:** Added explicit `dtype=np.float64` to the `np.sum` call: `np.sum(embeddings[:-1] * embeddings[1:], axis=1, dtype=np.float64)`. Inline comment documents the numpy version semantics.
- **Files modified:** `signal_processing.py.compute_deltas`.
- **Verification:** Test D (`da.dtype == np.float64`) passes; `_scores.npy` fixture is float64.
- **Committed in:** `5d3f3f4` (Task 1 atomic commit).

### Non-fix-required test inconsistencies (documented for Phase 3 awareness)

**Test M in Task 1 acceptance script is internally inconsistent with locked D-29 + Pitfall 10.**

The plan's acceptance script Test M used `sig = np.full(300, 0.01); sig[150] = 0.3` and expected `mad_normalize(sig)[150] >= 2.0`. With the locked `median_abs_deviation(scale=1.0)` raw MAD on a flat input plus a single spike, the local median absolute deviation is exactly `0.0` (179 of 180 samples are at the median), so the zero-MAD branch fires correctly per D-29 and emits `0.0`. The locked `<verify>` automated block does NOT include this test, and the test is inconsistent with the same D-29 behavior that Test L (300 zeros → all zeros) explicitly relies on. The implementation is correct per locked decisions; flagging here so a future reviewer doesn't try to "fix" the floor or the consistency factor.

## Threat Flags

None. The threat register's STRIDE row for `T-02-04` (argparse path construction) holds: `video_name` is only used as a stem under `output/cache/`, no subprocess in this phase, no untrusted-byte parsing.

## Phase 2 Success Criteria — Final Status (against ROADMAP §2 SC1–SC5)

| SC | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | First 20 raw deltas printed; length N-1; helper is single conversion site | PASS | `[deltas] first 20: [0.0421, 0.0404, ..., 0.0214]`; `[deltas] len=2268`; only one `score_index_to_timestamp` definition in source |
| 2 | Synthetic two-color test detects peak at K*0.5 within 0.01 s | PASS | `[alignment] PASS  peak at score-idx 6 → ts 3.5s (expected 3.5s)` (ε=1e-9, tighter than 0.01 s) |
| 3 | smooth_deltas removes spikes; no phantom edge dip | PASS | `[smoothed] spike-injection PASS  edge-preservation PASS` |
| 4 | MAD min/max/mean printed; max ≥ 2.0; <90% above 3.0; <5% zero-MAD | PASS | `[scores] min=0.0000 max=10.0000 mean=0.7577`; above 3.0 = 6.48%; zero-MAD = 0.00% |
| 5 | ruptures never imported when --pelt off; --pelt path returns list[int] from rpt.Pelt(model='rbf') | PASS | `[lazy-import] PELT clean`; with `--pelt`: `[pelt] 86 changepoints; first 10: [30, 50, ...]`; independent sys.modules probe PASS |

## Open issues for Phase 3

- Confirm `clip_selection.py` imports `score_index_to_timestamp` from `signal_processing` rather than reimplementing `timestamps[idx + 1]` inline (cross-cutting alignment invariant — would defeat Pitfall 8 mitigation).
- JT signal characteristics observed: max score ceiling-pegs at 10.0 in 6.48% of samples, mean 0.7577 — typical for a 19-min body cam with a few high-action segments. No retuning candidate flagged on JT.
- 86 PELT changepoints on JT smoothed signal at penalty=3.0 — Phase 5 will compare clip selections with/without `--pelt` boost on this fixture during parameter tuning.
- `mad_normalize` is a Python loop (~2,268 iterations). 0.49 s wall-clock on JT — fine. If Phase 6 batch hits the multi-video path and one video has N >> 5,000 frames, vectorization via stride tricks becomes a v2 perf candidate (deferred per CONTEXT).

## Self-Check: PASSED

- `signal_processing.py` exists at expected path: FOUND
- `output/cache/justin_timberlake_scores.npy` exists, shape `(2268,)`, dtype `float64`: FOUND
- `output/cache/justin_timberlake_changepoints.npy` exists, shape `(86,)`, dtype `int64` (after --pelt run): FOUND
- Commit `5d3f3f4` (Task 1): FOUND in `git log`
- Commit `baf798b` (Task 2): FOUND in `git log`
- §0.5 harness final stdout line `Phase 2 §0.5 verification: PASS`: REPRODUCED from `/tmp/sp_run.log`
- Lazy-import contract verified independently: PASS
