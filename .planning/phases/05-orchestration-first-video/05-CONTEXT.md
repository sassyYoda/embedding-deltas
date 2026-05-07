# Phase 5: Orchestration & First-Video End-to-End - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Mode:** `--auto`

<domain>
## Phase Boundary

Wire the four module phases (extract → signal_processing → clip_selection → export) into `pipeline.py`, emit the spec §8 JSON manifest, lock determinism env vars at process entry, and produce **one watchable reel** on the chosen representative video — Justin Timberlake. After Phase 5: parameter values for `--height`, `--min-gap-sec`, `--merge-gap-sec` are tuned and **frozen** for Phase 6's batch run on the remaining videos.

Phase 5 delivers (in `pipeline.py`):
- argparse with positional `video` + flags `--pelt`, `--height` (default 1.5), `--min-gap-sec` (default 15.0), `--merge-gap-sec` (default 3.0).
- `run(video_path, *, use_pelt, height, min_gap_sec, merge_gap_sec)` function that orchestrates all 6 stages, prints `[1/6] … [6/6]` banners, calls Phase 1–4 functions in order, assembles the JSON manifest, writes the reel.
- Determinism env-var stanza (`OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True, warn_only=True)`) at the top of `if __name__ == "__main__":` BEFORE any heavy imports — but with a small dance because torch must already be importable.
- JSON §8 manifest assembly with `mad_score`, `raw_cosine_delta`, `coincides_with_pelt_changepoint` per clip, and `embedding_model`, `sampling_fps`, etc. at the top level. Float rounding: 3dp seconds, 4dp scores.
- Two-run-byte-identical-JSON sanity check (recorded in SUMMARY).

**Not in this phase:** ffmpeg primitives (Phase 4), signal/clip math (Phases 2/3), CLIP inference (Phase 1), batch run on remaining 4 videos (Phase 6), README writeup (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### CLI / argparse

- **D-64:** `pipeline.py` argparse exposes EXACTLY the spec §9 surface:
  - Positional: `video` (path to a `.mp4`).
  - Flags: `--pelt` (boolean, default False), `--height` (float, default 1.5), `--min-gap-sec` (float, default 15.0), `--merge-gap-sec` (float, default 3.0).
  - `--help` text mirrors spec §9 verbatim — these flags are the user-facing tuning surface.
  - **NO additional flags** (no `--cache`, `--save-fixture`, `--diagnostics`, `--verbose`, `--device`). Those are dev-time concerns owned by individual modules' `__main__` blocks. Spec §9 is the locked CLI contract.
  - (Reference ORCH-01.)

### Pipeline `run()` Function

- **D-65:** `run(video_path: str, *, use_pelt: bool, height: float, min_gap_sec: float, merge_gap_sec: float) -> None` — keyword-only args after `video_path` for self-documentation. Returns None; all output is on disk + stdout.
- **D-66:** `run()` orchestrates EXACTLY 6 stages, with stage banners matching spec §9 verbatim:
  1. `[1/6] Sampling frames at 2fps...` → `extract.sample_frames(video_path, fps=2.0)` → `(frames, timestamps_list)`.
  2. `[2/6] Extracting CLIP embeddings...` → `extract.load_model()` (NOT cached across calls in v1; reload per video for simplicity in `run()`) → `extract.embed_frames(frames, model, preprocess, device)` → `embeddings`.
  3. `[3/6] Computing embedding deltas...` → `signal_processing.compute_deltas(embeddings)` → `raw_deltas`. **Save reference to `raw_deltas` for JSON's `raw_cosine_delta` field per clip.**
  4. `[4/6] Filtering and normalizing signal...` → `smooth_deltas(raw_deltas)` → `smoothed`. → `mad_normalize(smoothed)` → `scores`. If `use_pelt`: `detect_changepoints(smoothed)` → `changepoints` (else `None`).
  5. `[5/6] Selecting clips...` → `select_peaks(scores, timestamps, height, min_gap_sec)` → `(peak_indices, peak_scores)`. If `use_pelt`: `apply_pelt_boost(peak_indices, peak_scores, changepoints)` → reordered. → `compute_budget_seconds(duration)` → `budget`. → `compute_padding(budget)` → `padding`. → `build_clips(...)` → `merge_clips(..., merge_gap_sec)` → `enforce_budget(..., budget)` → `final_clips: list[(start, end, score, peak_time)]`.
  6. `[6/6] Exporting highlight reel...` → for each clip: `export.extract_clip(...)` to `output/clips/{video_name}/{NNN}.mp4`. → `export.concat_clips(...)` to `output/reels/{video_name}_highlight.mp4`. → assemble + write JSON to `output/timestamps/{video_name}.json`.
- **D-67:** Final `Done. Reel: output/reels/{video_name}_highlight.mp4` print.

### Determinism Env Vars (Pitfall 14)

- **D-68:** Determinism stanza is set **at the top of `if __name__ == "__main__":` block, BEFORE any other code in that block, BEFORE argparse**:
  ```python
  if __name__ == "__main__":
      import os
      os.environ.setdefault("OMP_NUM_THREADS", "1")
      os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

      import torch
      torch.use_deterministic_algorithms(True, warn_only=True)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

      # ... argparse + run() ...
  ```
  - **Critical:** `os.environ` MUST be set BEFORE `import torch` (CUBLAS_WORKSPACE_CONFIG is read at torch's CUDA initialization). The module-level `import torch` from `extract.py` happens transitively when `pipeline.py` does `from extract import ...`, so the entire `import` block of `pipeline.py` itself must happen INSIDE the `__main__` guard, OR the env vars must be set BEFORE the module-level imports.
  - **Recommended pattern** to avoid import-order bugs: set the env vars at the very top of `pipeline.py` (module level, BEFORE any import), then do all imports, then `if __name__ == "__main__":` for argparse + run.
  - `warn_only=True` on `use_deterministic_algorithms` is mandatory on MPS — strict mode raises on Apple Silicon (RESEARCH/STACK.md Mac/Apple Silicon section).
  - `setdefault` (not `=`) so user-set env vars are respected.
  - (Reference Pitfall 14, ROBU-02 v2 placement = Phase 5.)

### JSON §8 Manifest Assembly

- **D-69:** JSON schema follows spec §8 field-for-field:
  ```json
  {
    "video": "justin_timberlake.mp4",
    "video_duration_sec": 1134.144,
    "budget_sec": 37.805,
    "total_reel_duration_sec": 37.805,
    "embedding_model": "ViT-L-14 (OpenAI / QuickGELU)",
    "sampling_fps": 2.0,
    "clips": [
      {
        "clip_index": 0,
        "start_sec": 339.341,
        "end_sec": 350.682,
        "duration_sec": 11.341,
        "peak_timestamp_sec": 345.011,
        "mad_score": 6.5542,
        "raw_cosine_delta": 0.1873,
        "coincides_with_pelt_changepoint": null
      }
    ]
  }
  ```
  - `video`: basename (e.g., `justin_timberlake.mp4`).
  - `video_duration_sec`: from `utils.probe_video_metadata(...)["duration_sec"]`.
  - `budget_sec`: result of `compute_budget_seconds()`.
  - `total_reel_duration_sec`: sum of `(end - start)` across `final_clips`.
  - `embedding_model`: literal string `"ViT-L-14 (OpenAI / QuickGELU)"` — locked.
  - `sampling_fps`: literal `2.0` — locked.
  - (Reference JSON-01, JSON-02.)

- **D-70:** Per-clip computed fields:
  - `mad_score = float(scores[peak_idx])` — scores from Phase 2 (already MAD-normalized + clipped to [0,10]). The peak's index in score-space.
  - `raw_cosine_delta = float(raw_deltas[peak_idx])` — the original cosine delta at that index (Pitfall 6 / spec §4 — this is the unsmoothed value).
  - `peak_timestamp_sec = float(peak_time)` — the 4th element of the clip tuple (D-38 from Phase 3).
  - **Critical:** the `peak_idx` for a final clip is the score-array index that produced the clip's `peak_time`. To recover it: store `(peak_indices_post_boost, peak_scores_post_boost)` BEFORE building clips, then for each final clip find the original peak_idx via `np.argmin(np.abs(timestamps - peak_time)) - 1` (off-by-one helper from Phase 2). OR (simpler): change `clip_selection.build_clips` to return `(start, end, score, peak_time, peak_idx)` 5-tuples so Phase 5 doesn't have to reverse-resolve. **Recommendation: simpler path — pass `peak_idx` through the clip tuples in `pipeline.run()` only, NOT in clip_selection.py. Use a parallel dict `{peak_time: peak_idx}` constructed before `build_clips` and looked up after `enforce_budget`.**
  - `coincides_with_pelt_changepoint`:
    - When `use_pelt` is False → `None` (Python `None` → JSON `null`). Strict three-state rule (Pitfall 17).
    - When `use_pelt` is True → `True` if `peak_idx` is within `±5` samples of any element of `changepoints`, else `False`.
  - (Reference JSON-02, JSON-03.)

- **D-71:** Float rounding (Pitfall 16):
  - All `*_sec` fields rounded to **3 decimal places** (millisecond precision).
  - `mad_score` and `raw_cosine_delta` rounded to **4 decimal places**.
  - Round at JSON-assembly time (just before `json.dump`), not in the source ndarrays.
  - Use `round(value, n)` (banker's rounding is fine — same value across runs given identical inputs).
  - (Reference Pitfall 16, ROBU-03 v2 placement = Phase 5.)

- **D-72:** Strict three-state for `coincides_with_pelt_changepoint`:
  - `null` (Python `None`) when `--pelt` is OFF.
  - `true` / `false` (Python `True` / `False`) when `--pelt` is ON.
  - **NEVER omit the field. NEVER default to false when --pelt is off.** Static check in §0.5 acceptance: load the JSON, assert every clip has the key, assert the value is `None` when `--pelt` was off and `bool` when `--pelt` was on.
  - (Reference Pitfall 17, JSON-03.)

- **D-73:** JSON written via:
  ```python
  with open(json_path, "w") as f:
      json.dump(manifest, f, indent=2, sort_keys=False)  # preserve schema field order
      f.write("\n")  # trailing newline
  ```
  - `indent=2` for human readability.
  - `sort_keys=False` because spec §8 has a specific field order.
  - Trailing newline so files end cleanly.

### JSON Output Path

- **D-74:** `output/timestamps/{video_basename_stem}.json` (e.g., `justin_timberlake.json`). Created via `utils.ensure_output_dirs(stem)["timestamps"]`. (Reference JSON-01, ORCH-03.)

### Frozen Tuning Parameters (DOC-03)

- **D-75 (AMENDED 2026-05-06):** **Phase 5 tunes parameters on `marcus_jordan` (75.2 min) — the LONGEST video.** Spec §6 says verbatim: "Tune these once on the **longest/most representative** video, then hold them fixed across all five videos." Marcus Jordan is the longest by 2.7×, has 9,025 frames at 2 fps (4× the next-largest video), and is the most signal-rich — exactly what spec §6's tuning advice favors ("enough peaks to fill the budget greedily without so many that selection becomes arbitrary").
  - **Original D-75 chose JT in error** — the rationale "median is representative" contradicted the spec's explicit "longest/most representative" guidance. Corrected.
  - **`pipeline.py videos/justin_timberlake.mp4` is run as a SMOKE TEST** (validates code end-to-end: env-var stanza, JSON schema, two-run reproducibility) but the JT run is NOT the source of frozen tuning parameters.
  - **Tuning sequence:** (1) JT smoke test confirms pipeline.py works end-to-end. (2) Wait for marcus_jordan fixture (background streaming extract in progress). (3) Run `pipeline.py videos/marcus_jordan.mp4` and assess peak count, budget fill %, qualitative reel quality. (4) If parameters need adjustment from defaults `(--height=1.5, --min-gap-sec=15, --merge-gap-sec=3)`, sweep them on marcus_jordan. (5) Freeze the values that produced the best marcus_jordan reel.
- **D-76:** Initial parameter sweep on JT — try `(height, min_gap_sec, merge_gap_sec)` combinations on the cached embeddings (instant iteration via Phase 2/3 fixtures):
  - Default: `(1.5, 15.0, 3.0)` — produces 4 clips, fills budget 100%.
  - Variations to try: `(1.0, 15.0, 3.0)` (more candidate peaks), `(2.0, 15.0, 3.0)` (fewer/stronger peaks), `(1.5, 10.0, 3.0)` (finer min-gap), `(1.5, 15.0, 5.0)` (coarser merge).
  - Selection criterion: produced reel must have **diverse** clips (visually different scenes, not all from one stretch of the video) AND **fill** the budget close to 100% AND clips should **look like high-action moments** when watched. The last is a manual gate — Phase 5's `__main__` writes the reel to `output/reels/`, user plays it, assesses qualitatively.
- **D-77:** **Frozen parameters become the argparse defaults.** After tuning, if defaults need to change from `(1.5, 15.0, 3.0)`, update `argparse.add_argument(... default=NEW)` and document in SUMMARY.md the rationale ("tuned on JT, frozen for Phase 6"). If defaults are good, leave them as-is.
- **D-78:** Tuning is a **one-shot** — Phase 5 records the final values and Phase 6 invokes `pipeline.py` WITHOUT overriding them (just `python pipeline.py videos/{name}.mp4`). Per-video retuning is forbidden by spec §6 / §12. (Reference DOC-03.)

### Reproducibility Sanity Check

- **D-79:** After the main `run()` produces a reel + JSON, the harness performs a **two-run reproducibility test** by calling `run()` a second time on the SAME video with the SAME args, then `diff`-ing the two JSONs. They must be byte-identical. If they differ, document the diff in SUMMARY.md as a known nondeterminism source (likely MPS); it's NOT a hard blocker (spec doesn't mandate byte-identical reruns), but it informs Phase 6 batch behavior.
  - Implementation: temporarily move the first JSON aside, run again, diff, then move both back (keep both for inspection).
  - Acceptable variance per Pitfall 14: floats rounded to 3dp/4dp should be byte-stable; if they're not, the rounding isn't aggressive enough OR there's a non-determinism source we missed. Surface clearly.

### Module Imports

- **D-80:** `pipeline.py` imports:
  - stdlib: `os`, `argparse`, `json`, `pathlib`, `sys`, `time`.
  - Phases 1–4: `from extract import sample_frames, load_model, embed_frames`, `from signal_processing import compute_deltas, smooth_deltas, mad_normalize, detect_changepoints, score_index_to_timestamp`, `from clip_selection import select_peaks, apply_pelt_boost, compute_budget_seconds, compute_padding, build_clips, merge_clips, enforce_budget`, `from export import extract_clip, concat_clips`, `from utils import probe_video_metadata, ensure_output_dirs`.
  - **NO** direct `numpy`/`torch`/`cv2` imports in `pipeline.py` — those leak through extract.py's transitive imports but `pipeline.py` itself stays as a thin orchestrator. Phase 5 acceptance: `pipeline.py` source has `import torch` only inside the `if __name__ == "__main__":` block (for the env-var stanza per D-68); no other direct ML lib imports.

### Visual Sanity Check (Spec §0.5 Final Step)

- **D-81:** Spec §0.5 says: "Watch the original video (or at least scrub through it). Watch the reel. Ask: do the selected moments look like high-action or significant moments, or is the reel full of camera pans and walking?"
  - Phase 5's `__main__` cannot automate this. It MUST print the reel path and the original video path so the user can `open` both and compare.
  - The §0.5 print includes `[manual-check] play output/reels/{name}_highlight.mp4 vs videos/{name}.mp4 to qualitatively confirm`.
  - Phase 5 SUMMARY.md should record the user's qualitative verdict (manually-recorded by Claude after the user reports back) — this becomes the "JT tuning verdict" that Phase 6 README references.

### Multi-Video Awareness

- **D-82 (AMENDED 2026-05-06):** Phase 5 operates on **one video for tuning (marcus_jordan, the longest)** and **one video as a smoke test (JT, the first-processed)**. Phase 6's batch runs `pipeline.py` per video with the frozen parameters. The `run()` function is video-agnostic; spec §9 contract requires the positional `video` argument so there's no default-resolution logic in pipeline.py.

### Claude's Discretion

- Type hints, docstring style.
- Logger vs `print()` — stick with `print()` to match spec §9 snippet verbatim.
- Whether to wrap `run()` in a try/except for cleaner error messages or let exceptions propagate (let propagate; argparse handles SystemExit, run() propagates RuntimeError from extract/export).
- Numerical-safety epsilons for the two-run-diff (use byte-equality, not float-tolerance).

</decisions>

<canonical_refs>
## Canonical References

### Locked Spec
- `assignment-details/bodycam_highlight_reel_spec.md` §8 (JSON Schema), §9 (Main Pipeline Entry Point), §0.5 (testing protocol — final step is qualitative reel check).

### Inherited from Prior Phases
- `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` D-09 (`force_quick_gelu=True`), D-10 (`embed_frames(frames, model, preprocess)` signature), D-15 (Phase 5 owns the env-var stanza).
- `.planning/phases/02-signal-processing/02-CONTEXT.md` D-24 (`score_index_to_timestamp` helper).
- `.planning/phases/03-clip-selection/03-CONTEXT.md` D-38 (4-tuple `(start, end, score, peak_time)`).
- `.planning/phases/04-export/04-CONTEXT.md` D-58 (extract.py / signal_processing.py / clip_selection.py / export.py have their own `__main__` harnesses; pipeline.py's `__main__` is the unified entry).

### Project-Level
- `.planning/PROJECT.md` — Out of Scope: no `--cache`/`--diagnostics` flags in pipeline.py.
- `.planning/REQUIREMENTS.md` — REQ-IDs ORCH-01..04, JSON-01..04, DOC-03 (9 mapped to Phase 5).
- `.planning/ROADMAP.md` Phase 5 detail — 6 success criteria.

### Research
- `.planning/research/PITFALLS.md` §14 (determinism env vars), §16 (JSON float rounding 3dp/4dp), §17 (PELT three-state field, never omit). **Phase 5 owns these three.**
- `.planning/research/STACK.md` Mac/Apple Silicon section (MPS requires `warn_only=True` on `use_deterministic_algorithms`).
- `.planning/research/SUMMARY.md` cross-cutting alignment invariant; `--pelt` orthogonality.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- All Phase 1–4 functions are imported. Phase 5 adds NO new computational logic, only orchestration + JSON assembly + env vars.
- `output/cache/justin_timberlake_*.npy` exist but `pipeline.py` does NOT read these — it re-runs the full extraction. The cached `.npy` fixtures are dev-time only (CONTEXT 01 D-19, CONTEXT 02 D-35, CONTEXT 03 D-50).
- `output/reels/justin_timberlake_highlight.mp4` exists from Phase 4's harness. Phase 5's pipeline run will OVERWRITE it (different output may be byte-different — same source clip but pipeline.py's invocation might land on different keyframes vs export.py's harness).
- `output/clips/justin_timberlake/{000..003}.mp4` exist from Phase 4. Will be overwritten.

### Established Patterns

- **`if __name__ == "__main__":` for entry points** — same pattern as Phases 1–4, but pipeline.py is the user-facing unified CLI per spec §9.
- **`utils.ensure_output_dirs(stem)`** for `reels/`, `clips/`, `timestamps/`, `cache/` paths.
- **`utils.probe_video_metadata(path)["duration_sec"]`** for `video_duration_sec`.
- **Conventional commits** — `feat(05-NN-MM): subject`.

### Integration Points

- Phase 5 → user — the user runs `python pipeline.py videos/justin_timberlake.mp4` (or with flags). Output: reel + JSON + clips on disk.
- Phase 5 → Phase 6 — Phase 6's `run_all.sh` (or in-process loop) calls the same `pipeline.py` per video.

</code_context>

<specifics>
## Specific Ideas

- **D-68's import ordering** is the most subtle Phase 5 gotcha. The env vars for CUBLAS must be set BEFORE `import torch` runs (anywhere in the process). Since `from extract import ...` transitively imports torch, the SAFEST pattern is: set env vars at the very top of `pipeline.py` (lines 1–5, before ANY import). Then do all imports. Then `if __name__ == "__main__":` does argparse + run.
- **D-70's peak_idx recovery** is the most subtle JSON-assembly gotcha. The cleanest fix: between `select_peaks` and `build_clips`, build a `{peak_time: peak_idx}` dict (since `peak_time = timestamps[idx + 1]` is unique per peak in normal cases). After `enforce_budget`, look up each final clip's `peak_idx` via this dict. Phase 5's `run()` is the natural home for this dict — it doesn't belong in `clip_selection.py`.
- **D-79's two-run reproducibility test** — if it fails with non-trivial diffs, that's a real finding worth documenting. With QuickGELU + L2-normalize + 3dp/4dp rounding + determinism env vars, it SHOULD pass. If MPS introduces sub-millisecond float drift that survives rounding, the diff will be in `mad_score`/`raw_cosine_delta` only.
- **D-81's manual qualitative gate** is the project's success metric per spec §11 — there's no labeled GT, so "watch the reel and judge" IS the eval. Phase 5 cannot skip this; it surfaces the path and waits for user input.

</specifics>

<deferred>
## Deferred Ideas

- **`--diagnostics` flag** for plotting raw deltas / smoothed / MAD / peaks → v2 (DIAG-01).
- **Embedding cache for re-runs with different params** → v2 (DIAG-02). Phase 5's `run()` always re-extracts.
- **Model-load amortization across multiple videos** → Phase 6 concern (the in-process batch loop).
- **`--config <yaml>`** for parameter sweeps → out of scope.
- **JSON schema validation** (jsonschema library) → out of scope; the spec §8 schema is small enough to assert manually.
- **Logging library** (`logging` instead of `print`) → out of scope; `print` matches spec §9 verbatim.
- **Crash recovery** (resume from a partial run) → out of scope.
- **Per-video qualitative writeup** → Phase 6 (DOC-01).

</deferred>

---

*Phase: 05-orchestration-first-video*
*Context gathered: 2026-05-06*
*Mode: --auto*
