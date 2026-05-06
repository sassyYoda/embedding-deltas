# Phase 3: Clip Selection - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Mode:** `--auto` (recommended option auto-selected per gray area; choices logged in `03-DISCUSSION-LOG.md`)

<domain>
## Phase Boundary

Convert `(scores, timestamps)` into a final ordered list of clip tuples `(start_sec, end_sec, score, peak_time)` that fits the per-video duration budget, with index→seconds conversion **isolated to this module** via the `score_index_to_timestamp` helper imported from `signal_processing.py`, and with `peak_time` threaded through merging so JSON consumers can rely on `start_sec ≤ peak_timestamp_sec ≤ end_sec`.

Specifically Phase 3 delivers (in `clip_selection.py`):
- `select_peaks(scores, timestamps, height, min_gap_sec, fps=2.0) -> tuple[np.ndarray, np.ndarray]` — find peaks, sort by descending score; the indices returned are score-array indices (consumer uses `score_index_to_timestamp` for seconds).
- `apply_pelt_boost(peak_indices, peak_scores, changepoints, window=5, boost=1.2) -> tuple[np.ndarray, np.ndarray]` — opt-in score boost when `--pelt`.
- `compute_budget_seconds(video_duration_sec) -> float` — `(dur / 1800) * 60`.
- `compute_padding(budget_sec) -> float` — `max(3.0, min(8.0, budget * 0.15))`.
- `build_clips(peak_indices, peak_scores, timestamps, video_duration_sec, padding_sec) -> list[tuple]` — produces `(start, end, score, peak_time)` 4-tuples in chronological order.
- `merge_clips(clips, gap_threshold_sec=3.0) -> list[tuple]` — merge adjacent clips within gap; keep the higher score AND its associated `peak_time`; assert `start ≤ peak_time ≤ end` post-merge.
- `enforce_budget(merged_clips, budget_sec) -> list[tuple]` — greedy by score; partial clip when remainder ≥ 3s, **centered on `peak_time`** (Pitfall 18); final list re-sorted chronologically.
- `__main__` block — §0.5 verification harness against the JT scores fixture.

**Not in this phase:** ffmpeg cut/concat (Phase 4); JSON assembly (Phase 5); multi-video batch (Phase 6); CLI argparse (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Clip Tuple Structure (cross-cutting; impacts merge, JSON, Phase 5)

- **D-38:** Clips are represented as **4-tuples**: `(start_sec: float, end_sec: float, score: float, peak_time: float)`. The 4th element (`peak_time`) is mandatory through `build_clips → merge_clips → enforce_budget` so Phase 5's JSON assembly can populate `peak_timestamp_sec` per spec §8 with the correct alignment-invariant guarantee `start_sec ≤ peak_timestamp_sec ≤ end_sec`. (Pitfall 19; Phase 1's CONTEXT.md cross-cutting rule.)
- **D-39:** Tuples are immutable Python tuples (NOT dataclasses, NOT NamedTuples for v1). Spec §6 code uses raw tuples; matching it minimizes friction. v2 could refactor to NamedTuple if Phase 5/6 tests benefit. (Reference SELB-03..05.)

### Peak Detection

- **D-40:** `select_peaks(scores, timestamps, height=1.5, min_gap_sec=15.0, fps=2.0)`:
  - Calls `scipy.signal.find_peaks(scores, height=height, distance=int(min_gap_sec * fps))`. **`distance` is in samples, NOT seconds** (Pitfall 20). Add post-call assertion `min(np.diff(peaks)) >= int(min_gap_sec * fps)` to guard against version drift.
  - Sorts peaks by descending score: `sorted_idx = np.argsort(scores[peaks])[::-1]`.
  - Returns `(peaks[sorted_idx], scores[peaks[sorted_idx]])` — both `np.ndarray`.
  - **Note:** the `timestamps` parameter is accepted but only used in the §0.5 verification print (number of peaks + their timestamps via `score_index_to_timestamp`). The function body itself stays in score-index space until the very end; conversion to seconds happens in `build_clips`. (Reference SELP-01, SELP-02, SELP-04.)
- **D-41:** §0.5 verification print after `select_peaks`: print `[peaks] count=N timestamps=[ts1, ts2, ...] scores=[s1, s2, ...]`. The first 10 are sufficient if N is large. (Reference SELP-04.)

### PELT Boost (Opt-in)

- **D-42:** `apply_pelt_boost(peak_indices, peak_scores, changepoints, window=5, boost=1.2) -> tuple[np.ndarray, np.ndarray]`:
  - For each peak index, multiply its score by `boost` if any changepoint is within `±window` samples.
  - Re-sort descending by boosted score.
  - Returns `(reordered_indices, boosted_scores)`.
  - Pure index-space; takes no timestamps.
  - When `--pelt` is OFF, this function is **never called**. Phase 5's `pipeline.run()` orchestrates the call only when the `--pelt` flag is set. (Reference SELP-03; cross-cutting `--pelt` orthogonality.)

### Duration Budget & Padding

- **D-43:** `compute_budget_seconds(video_duration_sec)` returns `(video_duration_sec / 1800.0) * 60.0` per spec §6 verbatim. Returns float (no rounding here; rounding is Phase 5's JSON concern).
- **D-44:** `compute_padding(budget_sec, max_padding=8.0, min_padding=3.0)` returns `max(min_padding, min(max_padding, budget_sec * 0.15))` per spec §6 verbatim. (Reference SELB-01, SELB-02.)

### Clip Construction

- **D-45:** `build_clips(peak_indices, peak_scores, timestamps, video_duration_sec, padding_sec)`:
  - Uses `score_index_to_timestamp(idx, timestamps)` (imported from `signal_processing`) to get `peak_time`. **No inline `timestamps[idx + 1]` allowed.** The grep check `'timestamps[idx + 1]' not in src` AND `'timestamps[idx+1]' not in src` AND `'timestamps[i + 1]' not in src` is part of the §0.5 acceptance.
  - For each peak: `start = max(0.0, peak_time - padding_sec)`, `end = min(video_duration_sec, peak_time + padding_sec)`.
  - Returns `list[tuple[float, float, float, float]]` sorted **chronologically** by `start`.
  - (Reference SELB-03; alignment invariant.)

### Merge Logic

- **D-46:** `merge_clips(clips, gap_threshold_sec=3.0)`:
  - Iterate chronologically; merge if `start <= prev.end + gap_threshold_sec`.
  - When merging: extend `prev.end` to `max(prev.end, this.end)`, keep the **higher score** AND **the `peak_time` from the higher-scoring clip** (Pitfall 19 — peak_time must follow the score). If scores tie, keep the earlier peak_time (deterministic).
  - **Hard assertion at end of merge:** `for clip in merged: assert clip.start <= clip.peak_time <= clip.end` — guards against off-by-one in the merge logic. (Reference SELB-04; Pitfall 19.)
  - Returns `list[tuple[float, float, float, float]]` chronological.

### Budget Enforcement & Partial Clip

- **D-47:** `enforce_budget(merged_clips, budget_sec)`:
  - Sort by descending score (stable).
  - Greedy accumulate: for each clip in score order, if `total + duration <= budget_sec`, append; else compute remainder.
  - **Partial-clip rule (Pitfall 18):** if `remainder >= 3.0`, build a partial clip **centered on `peak_time`** instead of starting at `clip.start`:
    - `half = remainder / 2`
    - `partial_start = max(clip.start, peak_time - half)` — clamp to original clip's start
    - `partial_end = min(clip.end, partial_start + remainder)` — clamp to original clip's end
    - If `partial_end - partial_start < remainder * 0.95` (e.g., peak too close to clip edge), use `partial_start = clip.start, partial_end = clip.start + remainder` as fallback.
    - Append `(partial_start, partial_end, score, peak_time)`.
  - Re-sort chronologically by `start` for export-friendly output.
  - Hard assertion: `total ≤ budget_sec + 1e-6` (numerical safety).
  - Hard assertion: no two final clips overlap. (Reference SELB-05, SELB-06; Pitfall 18.)
- **D-48:** §0.5 verification print after `enforce_budget`: print each final clip as `[clip-N] start={start:.3f} end={end:.3f} dur={dur:.3f} score={score:.4f} peak_time={pt:.3f}`. Print `[budget] total={total:.3f}s / budget={budget_sec:.3f}s ({pct:.1%})`. (Reference SELB-06.)

### `__main__` Verification Harness

- **D-49:** Pattern matches Phase 2 D-34: `if __name__ == "__main__":` block, runnable as `python clip_selection.py [video_name] [--pelt] [--height H] [--min-gap-sec G] [--merge-gap-sec M]`.
  - Default `video_name`: first `*_scores.npy` in `output/cache/` (most recent), with `_scores` stripped → resolves to base.
  - Loads `output/cache/{video}_scores.npy` (Phase 2 fixture) AND `output/cache/{video}_timestamps.npy` (Phase 1 fixture).
  - Computes budget from `utils.probe_video_metadata(videos/{video}.mp4)["duration_sec"]` (single source of truth for duration).
  - Runs `select_peaks → [apply_pelt_boost if --pelt] → build_clips → merge_clips → enforce_budget`.
  - Prints D-41 + D-48 lines and `Phase 3 §0.5 verification: PASS` on success.
  - Exit codes: 0 success, 2 missing fixture, 3 assertion failure.
  - **Synthetic test embedded in `__main__`:** build a synthetic scores array with peaks at known indices, run the full pipeline, assert clip count and ordering match expected. (Catches off-by-ones in merge / budget logic that pure-fixture tests would miss.)

### Fixture Output

- **D-50:** `clip_selection.py --save-fixture` (default ON in `__main__`) writes `output/cache/{video}_final_clips.json` — a JSON list of dicts:
  ```json
  [{"start_sec": 142.5, "end_sec": 158.5, "score": 4.23, "peak_time": 150.5}, ...]
  ```
  This is a dev-time fixture for Phase 4 (`export.py`) to develop against without re-running Phases 1–3. **NOT spec §8 schema** — that's a Phase 5 concern with the full clips array + meta fields. (Reference: same `--save-fixture` pattern as Phases 1–2.)

### Multi-Video Awareness

- **D-51:** `clip_selection.py.__main__` operates on **one video at a time** (the CLI arg or default-resolved). Multi-video parameter sweeps for tuning are a Phase 5 concern. Phase 3 just produces per-video clip lists.

### Claude's Discretion

- Type-hint style (PEP 604 unions, lowercase generics).
- Whether `select_peaks`'s peak-sorting is `np.argsort(...)[::-1]` (matches spec §6 snippet) or `-np.argsort(-scores[peaks])` (slightly more idiomatic) — pick one consistently.
- `enforce_budget`'s sort stability: use `sorted(clips, key=lambda c: -c.score)` (Python's sort is stable) or `np.argsort(-scores)` — equivalent for our use, prefer the former for readability.
- Exact JSON formatting for fixture output (`indent=2` vs no indent — pick `indent=2` for readability since the fixture is a dev tool not on a hot path).
- Numerical-safety epsilons: `1e-6` for budget total assertion, `0.95` for the "peak too close to edge" fallback ratio.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked Spec
- `assignment-details/bodycam_highlight_reel_spec.md` §6 (Clip Selection — peak detection, padding, merge, budget), §0.5 (testing protocol), §10 (Key Design Decisions rows 7–11), §12 (no manual selection, no global threshold, no per-video retuning).

### Inherited from Prior Phases
- `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` D-13 (L2-norm asserted at extract; Phase 3 doesn't see embeddings), D-18 (`output/cache/{video}_*.npy` fixture pattern).
- `.planning/phases/02-signal-processing/02-CONTEXT.md` D-22 (deltas dtype float64), **D-24 (`score_index_to_timestamp` helper — Phase 3 imports this; do NOT reimplement)**, D-32 (PELT lazy-import — Phase 3 only calls `apply_pelt_boost` which receives the changepoints list, doesn't import `ruptures`).

### Project-Level
- `.planning/PROJECT.md` — Active requirements list; Out of Scope (no global threshold, no manual selection, no per-video retuning).
- `.planning/REQUIREMENTS.md` — REQ-IDs SELP-01..04, SELB-01..06 (10 mapped to Phase 3).
- `.planning/ROADMAP.md` Phase 3 detail — 5 success criteria.
- `.planning/STATE.md` — current memory.

### Research
- `.planning/research/SUMMARY.md` — cross-cutting timestamp-alignment invariant + `--pelt` orthogonality.
- `.planning/research/PITFALLS.md` §18 (partial-clip on peak), §19 (peak_time through merge), §20 (find_peaks distance unit). **Phase 3 owns these three pitfalls.**
- `.planning/research/ARCHITECTURE.md` — Phase 3 module data contract: `(scores, timestamps) → list[(start, end, score, peak_time)]`. Index→seconds conversion is the **single seam** here.

### External
- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html` — `distance` parameter is in samples, not seconds (Pitfall 20).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `output/cache/justin_timberlake_scores.npy` (2268,) float64 in `[0, 10]` — Phase 3 dev input.
- `output/cache/justin_timberlake_timestamps.npy` (2269,) float64 — used by `build_clips` via the `score_index_to_timestamp` helper.
- `output/cache/justin_timberlake_changepoints.npy` (when generated) — Phase 3 `apply_pelt_boost` consumes.
- `signal_processing.score_index_to_timestamp` — IMPORT this. Do NOT reimplement. Single conversion seam (D-24 in Phase 2).
- `utils.probe_video_metadata(video_path)["duration_sec"]` — single source of truth for video duration; Phase 3's `__main__` uses this.
- `clip_selection.py` is a stub (Plan 01-01 created it). Phase 3 replaces the stub.

### Established Patterns

- **Module `__main__` verification harness** (D-49) — same pattern as Phase 1 (D-16), Phase 2 (D-34).
- **Fixture artifact write side-effect** (D-50) — `--save-fixture` default-on in `__main__`; output to `output/cache/`; gitignored.
- **Index-space vs time-space separation** — Phase 2 enforced "no timestamps in module body". Phase 3 IS the conversion seam: `build_clips` is the FIRST function in the codebase to legitimately pair an index with a timestamp, and it does so via the imported helper.
- **Conventional commits** — `feat(03-NN-MM): subject` matching Phases 1 + 2.

### Integration Points

- **Phase 2 → Phase 3:** scores ndarray + timestamps ndarray + (optional) changepoints list. All via `np.load` from `output/cache/`.
- **Phase 3 → Phase 4:** `final_clips: list[tuple[float, float, float, float]]` consumed by `export.extract_clip` (one tuple per ffmpeg invocation) + `export.concat_clips`.
- **Phase 3 → Phase 5:** `pipeline.run()` orchestrates `select_peaks → (apply_pelt_boost) → compute_budget → compute_padding → build_clips → merge_clips → enforce_budget`. The fixture write in `__main__` is dev-time only; Phase 5 calls these functions directly without serializing.

</code_context>

<specifics>
## Specific Ideas

- **The 4-tuple peak_time threading (D-38) is the most subtle locked invariant in Phase 3.** A reviewer reading the merge logic will look for: "what happens to peak_time when two clips merge?" — the answer is "it follows the higher-scoring peak". A reviewer reading enforce_budget will look for: "what does a partial clip look like?" — the answer is "centered on peak_time, not starting at clip.start". Make these explicit in docstrings + inline comments.
- **The `start ≤ peak_time ≤ end` post-merge assertion (D-46)** is the single most important runtime guard in Phase 3. If it ever fires, Phase 5's JSON would emit a `peak_timestamp_sec` outside the clip bounds — silently wrong but easily caught by this assertion.
- **`scipy.signal.find_peaks` `distance` is in samples (D-40 / Pitfall 20).** The plan's static check should grep for `distance=` and assert it's only used with `int(min_gap_sec * fps)` (multiplied by fps). Inline the math in the call site so a reader sees the unit conversion.
- **Phase 3 has zero CLIP cost AND zero numpy heavy lifting.** The `__main__` harness on JT (2,268 scores → ~10–30 peaks → ~5–15 clips after merge → ≤ 1 minute of clip duration) should run in **well under 1 second**. If it takes longer, something is wrong (probably accidentally O(N²) in merge).

</specifics>

<deferred>
## Deferred Ideas

- **NamedTuple / dataclass for clips** — defer to v2; raw tuples match spec §6 verbatim and JSON serialization is trivial.
- **Configurable padding bounds (max=8, min=3)** — locked by spec §6; would only relax if a reviewer asks.
- **Configurable PELT boost factor (1.2)** — locked by spec §5; v2.
- **Multi-objective scoring** (e.g., diversity penalty for clips too close in semantic space) — research/FEATURES.md flagged as v3+; not in scope.
- **Beam search instead of greedy budget** — research/FEATURES.md anti-feature; greedy is the spec-mandated approach (§6).
- **Per-clip metadata in fixture JSON** (raw_cosine_delta, mad_score, coincides_with_pelt_changepoint) — those fields belong to Phase 5's spec §8 JSON output, NOT Phase 3's dev fixture. Phase 5 will compute them.

</deferred>

---

*Phase: 03-clip-selection*
*Context gathered: 2026-05-06*
*Mode: --auto (single pass; recommended option auto-selected per gray area)*
