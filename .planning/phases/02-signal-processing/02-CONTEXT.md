# Phase 2: Signal Processing - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Mode:** `--auto` (recommended option auto-selected per gray area; choices logged in `02-DISCUSSION-LOG.md`)

<domain>
## Phase Boundary

Convert the Phase 1 embedding fixture into a clean per-sample anomaly-score array in **index space**, with the timestamp-alignment invariant locked behind a single helper and verified by a synthetic two-color-video test, and with PELT changepoint detection available as an opt-in path that does not perturb the default pipeline.

Specifically Phase 2 delivers (in `signal_processing.py`):
- `compute_deltas(embeddings: np.ndarray) -> np.ndarray` — `(N, 768) → (N-1,)` cosine distance.
- `smooth_deltas(raw: np.ndarray) -> np.ndarray` — 5-tap median filter with edge-preserving boundaries.
- `mad_normalize(smoothed: np.ndarray, window_samples: int = 180) -> np.ndarray` — rolling MAD over centered 90 s window, clipped to `[0, 10]`.
- `detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0) -> list[int]` — opt-in PELT, **lazy-imports `ruptures` inside the function**.
- `score_index_to_timestamp(score_idx: int, timestamps: np.ndarray) -> float` — the single conversion helper; `timestamps[score_idx + 1]`.
- `__main__` block — §0.5 verification harness against the cached JT fixture; runs the synthetic-alignment test; writes `output/cache/{video}_scores.npy` (and optionally `_changepoints.npy` when `--pelt`).

**Not in this phase:** peak detection, clip building, merge, budget enforcement (Phase 3); ffmpeg cut/concat (Phase 4); JSON assembly + determinism env vars (Phase 5); multi-video batch + README (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Cosine Delta Computation

- **D-22:** `compute_deltas(embeddings)` returns `(N-1,)` array of cosine distances `1 - dot_product`. Implementation matches spec §4 verbatim: dot products via `np.sum(embeddings[:-1] * embeddings[1:], axis=1)` (correct because embeddings are L2-normalized — Phase 1's D-13 hard-asserts this, so we don't re-normalize here), then `np.clip(..., -1.0, 1.0)` for numerical safety, then `1.0 - clipped`. Output dtype `float64` (numpy default for `np.sum` on float32 input is float64; we keep that — premature to force float32). (Reference SIGD-01.)
- **D-23:** Output of `compute_deltas` is in **index space only**. The function does NOT take or return timestamps. Callers (Phase 3, Phase 5) use `score_index_to_timestamp()` for the index→seconds conversion. (Reference SIGD-02 + cross-cutting alignment invariant.)

### Timestamp Alignment Invariant (cross-cutting)

- **D-24:** A single helper function exists in `signal_processing.py`:
  ```python
  def score_index_to_timestamp(score_idx: int, timestamps: np.ndarray) -> float:
      """Convert a score-array index to its corresponding sampled-frame timestamp.

      Score[i] / smoothed[i] / delta[i] all align to timestamps[i+1] —
      the delta represents change ARRIVING AT frame i+1 (spec §4 closing note).
      """
      return float(timestamps[score_idx + 1])
  ```
  This is the **only** site in the codebase where score indices convert to seconds. `clip_selection.py` (Phase 3) imports and uses this function; `pipeline.py` (Phase 5) does not handle the conversion directly. `signal_processing.py` itself never sees `timestamps` — it works purely in index space.
- **D-25:** Synthetic alignment test (constitutes the answer to spec §4's "this is the natural choice — verify"):
  - Build a `(K, 768)` embedding fixture in-memory: rows 0..K-1 are unit-vector pointing to `[1, 0, 0, ...]`; rows K..end are unit-vector pointing to `[0, 1, 0, ...]`. Cosine delta is 0 except at index K-1 where it's 1.0.
  - Run `compute_deltas → smooth_deltas → mad_normalize` on this fixture with synthetic timestamps `np.arange(N) * 0.5`.
  - Assert `np.argmax(scores) == K - 1` AND `score_index_to_timestamp(K - 1, timestamps) == K * 0.5` (within ε=1e-9).
  - Test lives in `signal_processing.py.__main__` (D-30 verification harness). Not a separate pytest file (matches Phase 1 D-16). (Reference SIGD-02.)
- **D-26:** §0.5 verification at end of `compute_deltas`: print first 20 raw delta values; assert `len(deltas) == N - 1`; assert all deltas are in `[0.0, 2.0]` (cosine distance bound after `np.clip`). Print `[deltas] first 20: {arr}; len={N-1}`. (Reference SIGD-03.)

### Median Filter (Smoothing)

- **D-27:** Use `scipy.ndimage.median_filter(raw_deltas, size=5, mode='reflect')`, **NOT** `scipy.signal.medfilt(raw_deltas, kernel_size=5)`. Both are kernel-5 medians; the difference is boundary handling:
  - `scipy.signal.medfilt` zero-pads edges → produces phantom dips in the first/last 2 samples (Pitfall 9).
  - `scipy.ndimage.median_filter(mode='reflect')` reflects the signal at boundaries → preserves edge values, no phantom artifacts.
  - This is a **deviation from spec §5 verbatim** (which calls `medfilt`). Justification: research/PITFALLS.md §9 documents the exact failure mode; reflected boundaries match the "edge-preserving" intent the spec invokes when justifying median-over-Gaussian. The deviation is documented in the function docstring with the spec citation. (Reference SIGS-01, research/PITFALLS.md §9.)
- **D-28:** §0.5 verification at end of `smooth_deltas`: print min/max/mean of `(smoothed - raw)` to confirm filter effect; assert no isolated single-sample spikes survive (synthetic test: inject a `+0.5` spike at index 100 in a flat-zero signal, confirm `smoothed[100] == 0`). (Reference SIGS-02.)

### MAD Normalization

- **D-29:** `mad_normalize(smoothed, window_samples=180)` follows spec §5 step 2 except for one numerical guard:
  - Centered window: `[max(0, i - half), min(N, i + half)]` where `half = window_samples // 2 = 90`.
  - Local median: `np.median(local)`. MAD: `scipy.stats.median_abs_deviation(local)` (NOT `np.median(np.abs(local - median))` — `median_abs_deviation` applies the consistency factor 1.4826 by default; we want the **raw** MAD not normalized to std-equivalent → pass `scale='normal'`? **No, pass nothing** — the default `scale=1.0` returns raw MAD which matches spec §5 intent of "number of MADs above local median".
  - Wait — actually: `scipy.stats.median_abs_deviation` default in scipy 1.16 is `scale=1.0` (raw MAD). Verify in implementation. If the default has changed, pin `scale=1.0` explicitly. The plan should call this out.
  - **MAD-zero guard: `if mad > 1e-3` (NOT `> 1e-8` as spec §5 says).** Pitfall 10 — `1e-8` is too lenient; static-footage windows produce MAD values in `[1e-7, 1e-3]` that round-trip through float math and produce ceiling-pegged scores from divisions by near-zero. `1e-3` is a defensible floor; below it we emit `0.0`. Documented deviation. (Research/PITFALLS.md §10.)
  - Output clipped to `[0.0, 10.0]` per spec §5.
- **D-30:** §0.5 verification at end of `mad_normalize`: print `min/max/mean` of normalized scores; assert `max ≥ 2.0` (window not too large; if max < 2.0, the signal is "too flat" — spec §0.5 warning); assert `<90% of samples are above 3.0` (window not too small); also print percentage of zero-MAD samples (samples where the guard fired) — should be `<5%` on a typical body cam video. (Reference SIGM-04, research/PITFALLS.md §10.)
- **D-31:** Output dtype: `float64` (scipy default, matches deltas dtype from D-22). No premature float32 conversion. (Phase 3 and Phase 5 handle final JSON rounding.)

### PELT Changepoints (Opt-in)

- **D-32:** `detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0) -> list[int]`:
  - **Lazy import** at the top of the function body: `import ruptures as rpt`. Module-level import of `ruptures` is FORBIDDEN in `signal_processing.py` — when `--pelt` is off, `ruptures` should never be loaded. Verified in §0.5 by `assert 'ruptures' not in sys.modules` at end of a non-PELT run.
  - Implementation matches spec §5 verbatim:
    ```python
    model = rpt.Pelt(model="rbf").fit(smoothed.reshape(-1, 1))
    changepoints = model.predict(pen=penalty)[:-1]  # drop trailing len(signal)
    return changepoints
    ```
  - Returns a list of integer indices (positions in `smoothed`).
- **D-33:** PELT is plumbed via the `--pelt` boolean flag at the CLI level (Phase 5 owns argparse); Phase 2 just provides `detect_changepoints()` and never auto-invokes it. (Reference SIGP-01..03, cross-cutting `--pelt` orthogonality from CONTEXT 01.)

### `__main__` Verification Harness

- **D-34:** Pattern matches Phase 1 (D-16): `if __name__ == "__main__":` block at the bottom of `signal_processing.py`, runnable as `python signal_processing.py [video_name]`.
  - Default arg: first `*_embeddings.npy` filename in `output/cache/` (most recent), with stem stripped of `_embeddings` suffix → resolves to base `<video_name>` for output filenames.
  - Steps:
    1. Load `output/cache/{video_name}_embeddings.npy` and `_timestamps.npy`.
    2. Run `compute_deltas` → emit §0.5 print 1 (D-26).
    3. Run `smooth_deltas` → emit §0.5 print 2 + spike-injection synthetic (D-28).
    4. Run `mad_normalize` → emit §0.5 print 3 (D-30).
    5. Run synthetic alignment test (D-25). On pass, print `[alignment] PASS`.
    6. Verify `'ruptures' not in sys.modules` → print `[lazy-import] PELT clean`.
    7. Optionally — if `--pelt` flag set in argparse — call `detect_changepoints(smoothed)`, print count + first 10 changepoints, then `assert 'ruptures' in sys.modules`.
    8. Write fixture: `output/cache/{video_name}_scores.npy` (float64). When `--pelt`, also write `output/cache/{video_name}_changepoints.npy` (int64).
    9. Print `Phase 2 §0.5 verification: PASS`.
  - Exit 0 on success, 2 on missing fixture, 3 on assertion failure.

### Fixture Output (for Phase 3 dev iteration)

- **D-35:** `signal_processing.py --save-fixture` (default ON in `__main__`) writes:
  - `output/cache/{video_name}_scores.npy` — `(N-1,) float64`, post-MAD-normalized + clipped scores.
  - `output/cache/{video_name}_changepoints.npy` — only if `--pelt` was passed; `(M,) int64` indices into `smoothed`.
  - These are dev-time fixtures (gitignored under `output/`); Phase 3 development against them is fast (~seconds, no CLIP).
  - **No `--save-fixture` flag in `pipeline.py`**; that's Phase 5's argparse and stays clean. Same pattern as Phase 1's D-18 / D-19.

### Multi-Video Awareness

- **D-36:** `signal_processing.py.__main__` operates on **one video at a time** (the CLI arg or the most-recent fixture). Multi-video signal comparison (e.g., picking the representative video for parameter tuning) is a Phase 5 concern; Phase 2 just produces per-video fixtures that Phase 5 can read.
- **D-37:** A background sequential extract loop (the user kicked it off concurrently with Phase 2 planning) is producing fixtures for `marcus_jordan`, `tiger_woods`, `test_assault_theft`, `test_missing_person` over ~52 minutes. Phase 2 dev does NOT block on those — JT fixture is sufficient for §0.5 + synthetic-alignment verification. As fixtures land, the user can re-run `python signal_processing.py <video>` to spot-check signal distributions, but that's optional.

### Claude's Discretion

- Internal docstring style, type-hint style (PEP 604, lowercase `dict[k,v]`), error message wording.
- Whether to use `np.argmax` or argpartition in the synthetic alignment test (use `np.argmax` — clearer intent, perf irrelevant on K-element array).
- Tolerance constants: `ε=1e-9` for synthetic alignment test, `1e-3` MAD floor (locked by D-29), `[0, 2]` raw delta bounds (locked by cosine math).
- Whether `mad_normalize` is implemented as a Python loop (spec §5 snippet) or vectorized via stride tricks. Default to the loop — matches spec verbatim, only ~2,300 iterations on JT (one per score), trivial cost. Vectorization is a v2 perf optimization.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, executor) MUST read these before planning or implementing.**

### Locked Spec
- `assignment-details/bodycam_highlight_reel_spec.md` §4 (Delta Computation), §5 (Signal Processing — median filter + MAD + PELT), §0.5 (Testing Protocol — per-module prints), §10 (Key Design Decisions row 4–6 + 12), §12 (What Not To Do — no global threshold, no raw-frame median filter).

### Phase 1 (locked decisions inherited)
- `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` D-13 (L2 normalization is hard-asserted at extract time — Phase 2 does NOT re-normalize), D-15 (determinism precursors — Phase 2 inherits, no env vars), D-18 (fixture artifacts at `output/cache/{video_name}_*.npy`).

### Project-Level
- `.planning/PROJECT.md` — Core Value, Constraints, Key Decisions table (rows on median filter, MAD window, PELT opt-in).
- `.planning/REQUIREMENTS.md` — REQ-IDs SIGD-01..03, SIGS-01..02, SIGM-01..04, SIGP-01..03 (12 mapped to Phase 2).
- `.planning/ROADMAP.md` Phase 2 detail — 5 success criteria.
- `.planning/STATE.md` — current memory.

### Research
- `.planning/research/SUMMARY.md` — "Cross-cutting invariant" + "Phase 2 in build order" sections.
- `.planning/research/PITFALLS.md` §8 (alignment off-by-one), §9 (medfilt edge effects → use `ndimage.median_filter(mode='reflect')`), §10 (MAD floor `1e-3` not `1e-8`). **Phase 2 owns these three pitfalls.**
- `.planning/research/ARCHITECTURE.md` — Phase 2 module data contract: `embeddings (N, 768) → scores (N-1,) float64`, no timestamps in this module.

### External
- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html` — for `mode='reflect'` semantics (D-27).
- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html` — for `scale=1.0` default and `nan_policy='propagate'` (D-29).
- `https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/` — for `Pelt(model="rbf")` semantics (D-32).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `output/cache/justin_timberlake_embeddings.npy` (2269, 768) float32 L2-normalized — primary Phase 2 dev input.
- `output/cache/justin_timberlake_timestamps.npy` (2269,) float64 — used only by `clip_selection.py` (Phase 3) and the Phase 2 §0.5 alignment test.
- `extract.py.__main__` block — Phase 2 should mirror this pattern verbatim (D-34).
- `utils.ensure_output_dirs(name)["cache"]` — fixture write target (already used by extract.py).
- `signal_processing.py` exists as a stub (Plan 01-01 created it with a placeholder docstring). Phase 2's first task is to replace the stub with the real implementation.

### Established Patterns

- **Index-space-only computation:** `signal_processing.py` will be the FIRST enforcer of the index-space rule. No timestamps inside the module body; only the synthetic-alignment test in `__main__` and the `score_index_to_timestamp()` helper touch them.
- **§0.5 verification gate:** module-level `__main__` block, runnable as `python <module>.py`. Same as `extract.py`.
- **Lazy-import opt-in deps:** `ruptures` lazy-imported inside `detect_changepoints()`. First instance of this pattern in the project.
- **Fixture artifact write side-effect:** `--save-fixture` (or implicit-on-default in `__main__`) writes intermediate `.npy` files to `output/cache/` for Phase 3 dev. Same as Phase 1.

### Integration Points

- **Phase 1 → Phase 2:** `embeddings` ndarray flows in via `np.load(output/cache/{video}_embeddings.npy)`.
- **Phase 2 → Phase 3:** `scores` ndarray + `score_index_to_timestamp()` helper. Phase 3 imports the helper from `signal_processing` rather than reimplementing the index→seconds conversion.
- **Phase 2 → Phase 5:** `pipeline.py` will call `compute_deltas → smooth_deltas → mad_normalize` (and optionally `detect_changepoints`) on a freshly-extracted embedding array. The fixture writes are dev-time only and bypassed in `pipeline.run()`.

</code_context>

<specifics>
## Specific Ideas

- **Synthetic alignment test (D-25)** is THE answer to spec §4's "natural choice — verify". A reviewer reading Phase 2 will look for evidence that the alignment was actually checked, not just claimed. Make the test prominent in `__main__` output.
- **D-27 deviation from spec §5 verbatim** (using `ndimage.median_filter` instead of `signal.medfilt`): document in the function docstring explicitly with `# Deviation from spec §5: ...` so reviewers see the rationale inline. Same for D-29's `1e-3` MAD floor.
- **Phase 2 has zero CLIP cost.** Pure numpy/scipy on a pre-computed embedding fixture. The §0.5 harness should run in **under 5 seconds** on JT's 2,269-frame fixture. If it takes longer, something is wrong (probably accidentally-vectorized over the whole signal).
- **PELT lazy-import verification (D-32)** uses `assert 'ruptures' not in sys.modules` at end of non-PELT run. This is testable via `python -c "import sys; import signal_processing; sp.compute_deltas(...); ...; assert 'ruptures' not in sys.modules"`. Make sure the lazy-import is INSIDE the function body, not at module top with a try/except.

</specifics>

<deferred>
## Deferred Ideas

- **Vectorized rolling-MAD via stride_tricks** — perf optimization; trivial cost on JT, defer until/unless Phase 6 batch processing reveals it. v2 ROBU concern.
- **Multi-video signal comparison** — Phase 5 concern; Phase 2 produces per-video fixtures, doesn't analyze cross-video.
- **Configurable median kernel size / MAD window** — locked at 5 / 180 by spec §5; would only relax if a reviewer asks. v2.
- **Custom PELT model (cosine instead of rbf)** — research/SUMMARY.md flagged this as low-priority. v2.
- **Save smoothed-deltas as separate fixture** — only `scores` is saved (D-35); `smoothed` is intermediate. If Phase 3 ever needs `smoothed` directly, surface as a small extension; for now it's regenerated trivially from `compute_deltas + smooth_deltas`.

</deferred>

---

*Phase: 02-signal-processing*
*Context gathered: 2026-05-06*
*Mode: --auto (single pass; recommended option auto-selected per gray area)*
