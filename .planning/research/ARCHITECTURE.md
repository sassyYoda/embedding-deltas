# Architecture Research

**Domain:** Single-process Python CLI pipeline — video → CLIP embeddings → 1D signal processing → peak/clip selection → ffmpeg export
**Researched:** 2026-05-06
**Confidence:** HIGH (spec is locked and prescriptive; this document validates and tightens it rather than redesigning it)

---

## 0. TL;DR

- The spec's 5-module + `pipeline.py` orchestrator split is **the right granularity**. Do not refactor it.
- One refinement worth making: **`utils.py` is real, not optional** — it owns video metadata probing, output-directory creation, and the JSON writer, all of which are touched by ≥2 modules and would otherwise create subtle import cycles.
- The data flow is a **strict linear DAG** with one branch (PELT). All cross-module contracts are plain numpy arrays + lists of tuples + Python primitives — no shared mutable state, no classes required.
- Build order from the spec footer (`extract → signal_processing → clip_selection → export → pipeline`) is **dependency-correct but not maximally parallel**. With one researcher writing fixtures from `extract.py` early, `signal_processing.py`, `clip_selection.py`, and `export.py` can be built **in parallel** off recorded fixtures. See §6.
- The `--pelt` branch lives in **both** `signal_processing.py` (detection) **and** `clip_selection.py` (score boost + JSON field) — never in `pipeline.py`'s logic, only in its argparse.
- All output artifacts are written by **`export.py`** (reels + clips) and **`pipeline.py`** (timestamps JSON). `utils.py` owns directory creation. `extract.py` and `signal_processing.py` write nothing to disk in production runs (they may write debug artifacts during §0.5 verification, but those are not load-bearing).

---

## 1. System Overview

```
                        ┌───────────────────────────────┐
                        │         pipeline.py           │
                        │  argparse + run() orchestrator│
                        │  prints [1/6]…[6/6] progress  │
                        └───────────────┬───────────────┘
                                        │ calls
        ┌───────────────────────────────┼────────────────────────────────┐
        │                               │                                │
        ▼                               ▼                                ▼
┌──────────────┐             ┌──────────────────────┐          ┌─────────────────┐
│  extract.py  │             │ signal_processing.py │          │ clip_selection  │
│              │             │                      │          │      .py        │
│ sample_frames│             │ compute_deltas       │          │ select_peaks    │
│ embed_frames │             │ smooth (medfilt)     │          │ apply_pelt_boost│
│              │             │ mad_normalize        │          │ compute_padding │
│              │             │ detect_changepoints* │          │ compute_budget  │
│              │             │   (*--pelt only)     │          │ build_clips     │
│              │             │                      │          │ merge_clips     │
│              │             │                      │          │ enforce_budget  │
└──────┬───────┘             └──────────┬───────────┘          └────────┬────────┘
       │                                │                               │
       │ frames, timestamps,            │ scores, [changepoints],       │ final_clips
       │ embeddings                     │ raw_deltas (for JSON)         │ list[(s,e,score)]
       ▼                                ▼                               ▼
                              [in-memory only — no I/O]
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │      export.py       │
                                                          │ extract_clip (×N)    │
                                                          │ concat_clips         │
                                                          └──────────┬───────────┘
                                                                     │ writes
                                                                     ▼
                                            output/clips/{video_name}/clip_*.mp4
                                            output/reels/{video_name}_highlight.mp4

       ┌────────────────────────────────────────────────────────────────────┐
       │                            utils.py                                │
       │  probe_video_metadata(path) → (duration_sec, fps, w, h)            │
       │  ensure_output_dirs(video_name)                                    │
       │  write_timestamps_json(video_name, payload)                        │
       │  log_step(idx, total, msg)         ← used by pipeline.py prints    │
       └────────────────────────────────────────────────────────────────────┘
                  ↑ imported by pipeline.py, export.py, extract.py
```

**Asterisks** (`*`) mark `--pelt`-conditional code paths.

---

## 2. Module Boundaries — Validation of Spec §1

The spec locks 5 modules + `pipeline.py`. Each is justified below; none should be split or merged.

| Module | Responsibility | Why this boundary is correct |
|---|---|---|
| `extract.py` | Video → frames → embeddings | Owns the only two heavyweight third-party deps: `cv2` for I/O and `open_clip` + `torch` for inference. Isolating these here means signal-processing tests don't pull in `torch`. |
| `signal_processing.py` | embeddings → raw_deltas → smoothed → scores → (optional changepoints) | Pure numpy/scipy; deterministic; no I/O. The whole module is fixture-testable in isolation, which is exactly what §0.5 demands. |
| `clip_selection.py` | scores + timestamps → ranked peaks → padded clips → merged → budgeted | Pure Python + scipy. Decoupled from how scores were computed and from how clips are exported. |
| `export.py` | clip list → on-disk clip files → concatenated reel | Owns the only `subprocess` calls to `ffmpeg`. Isolating shells-out here means the rest of the pipeline is pure-Python and unit-testable. |
| `utils.py` | shared helpers (video metadata, directory setup, JSON writer, logging) | Prevents import cycles: `pipeline.py` and `export.py` both need video duration; `pipeline.py` and `utils.py` both need output paths. |
| `pipeline.py` | argparse + linear orchestration + JSON assembly | Thin glue. Should contain **no algorithmic logic** — only function calls and the `run()` function. |

**Refinements to the spec (none structural, all clarifying):**

1. **`utils.py` is load-bearing, not vestigial.** The spec lists it but never specifies its contents. It must own at minimum:
   - `probe_video_metadata(video_path) → dict` — single source of truth for `video_duration_sec`, `video_fps`, frame count. Called once at the top of `pipeline.run()`; result threaded through to everyone who needs it. Avoids `cv2.VideoCapture` opening the file twice.
   - `ensure_output_dirs(video_name) → dict[str, Path]` — creates `output/reels`, `output/clips/{video_name}`, `output/timestamps`; returns paths. Called by `pipeline.run()` before any work.
   - `write_timestamps_json(path, payload)` — atomic write of the §8 schema. Owned here because `pipeline.py` should not contain `json.dump` boilerplate inline.
2. **Diagnostics are local to each module, not centralized.** §0.5 specifies per-module verification prints. These belong inside the respective module (e.g. `extract.py` prints frame count + first 5 timestamps + embedding shape; `signal_processing.py` prints raw delta head + score min/max/mean). Gate them behind a `verbose=True` kwarg on each public function — **don't** route them through `utils.log_step()`. The progress prints `[1/6]…[6/6]` are the only thing `pipeline.py` prints.
3. **Per the spec's "When `--pelt` is not active, `ruptures` does not need to be imported"** — this is enforceable only if `detect_changepoints` is imported lazily inside the function or wrapped behind an `if use_pelt:` guard at call-site in `pipeline.py`. Top-level `import ruptures` in `signal_processing.py` violates this. Recommended: import `ruptures` inside `detect_changepoints()` so the import only fires when the function is called.

---

## 3. Data Contracts (Inter-Module Types)

Every arrow in the diagram has an explicit type. These are the contracts roadmap phases must honor.

### 3.1 `extract.py` outputs

```python
# sample_frames(video_path: str, fps: float = 2.0) returns:
frames:     list[np.ndarray]   # length N; each (H, W, 3) uint8 in RGB
timestamps: np.ndarray         # shape (N,) float64, units = seconds, monotonically increasing,
                               # spaced ≈ 1/fps = 0.5s apart (drift acceptable, see §5.1)

# embed_frames(frames, model, preprocess) returns:
embeddings: np.ndarray         # shape (N, 768) float32, L2-normalized along axis=1
                               # invariant: np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
```

**Critical:** `frames` is a list, not a stacked array, because per-frame shape may vary slightly if the source has resolution glitches and because stacking 3,600×H×W×3 uint8 in RAM is wasteful — `embed_frames` should batch-and-discard, never holding all frames as one tensor. After `embed_frames` returns, `frames` can be dropped (`del frames`) before the next stage to free memory.

### 3.2 `signal_processing.py` outputs

```python
# compute_deltas(embeddings) returns:
raw_deltas: np.ndarray         # shape (N-1,) float32, range [0, 2], typical [0.0, 0.5]
                               # raw_deltas[i] aligns to timestamps[i+1] (delta arriving AT frame i+1)

# smooth_deltas(raw_deltas, kernel_size=5) returns:
smoothed: np.ndarray           # shape (N-1,) float32, same alignment as raw_deltas

# mad_normalize(smoothed, window_samples=180) returns:
scores: np.ndarray             # shape (N-1,) float32, clipped to [0, 10]
                               # scores[i] aligns to timestamps[i+1]

# detect_changepoints(smoothed, penalty=3.0) returns (only when --pelt):
changepoints: list[int]        # indices into the (N-1,) signal, excluding the trailing len(signal)
                               # empty list if PELT finds no breakpoints
```

**Alignment convention (NON-NEGOTIABLE):** All three (`raw_deltas`, `smoothed`, `scores`) have length `N-1` and use the same alignment: index `i` corresponds to `timestamps[i+1]`. `clip_selection.py` relies on this when converting peak indices to seconds.

### 3.3 `clip_selection.py` outputs

```python
# select_peaks(scores, height, min_gap_sec, fps) returns:
peak_indices: np.ndarray       # shape (P,) int64, indices into `scores`, sorted by score desc
peak_scores:  np.ndarray       # shape (P,) float32, the scores at those indices, sorted desc

# apply_pelt_boost(peak_indices, peak_scores, changepoints, window=5, boost=1.2) returns:
peak_indices: np.ndarray       # shape (P,) int64, re-sorted by boosted score desc
peak_scores:  np.ndarray       # shape (P,) float32, BOOSTED values where applicable, sorted desc

# build_clips(peak_indices, peak_scores, timestamps, video_duration_sec, padding_sec) returns:
clips: list[tuple[float, float, float]]
                               # each = (start_sec, end_sec, score), sorted chronologically by start
                               # invariant: 0 <= start < end <= video_duration_sec

# merge_clips(clips, gap_threshold_sec=3.0) returns:
merged: list[tuple[float, float, float]]
                               # same shape, fewer entries; non-overlapping; chronological

# enforce_budget(merged, budget_sec) returns:
final_clips: list[tuple[float, float, float]]
                               # sum(end - start) <= budget_sec; chronological
```

### 3.4 `export.py` outputs

```python
# extract_clip(input_path, output_path, start_sec, end_sec) returns:
None                           # side effect: writes output_path; raises CalledProcessError on failure

# concat_clips(clip_paths, output_path, temp_dir) returns:
None                           # side effect: writes output_path

# Suggested wrapper:
# export_reel(video_path, final_clips, video_name, paths) → list[Path]
#   returns the list of intermediate clip paths so pipeline.py can include them
#   in the JSON if desired, and so they can be cleaned up.
```

### 3.5 `pipeline.py` outputs

```python
# Writes ONE artifact directly: the JSON manifest.
# Schema is locked by spec §8; all fields are filled from values already computed above.
# `coincides_with_pelt_changepoint` is null when use_pelt=False, bool otherwise.
```

### 3.6 What flows through `pipeline.run()`

`pipeline.py` is the only place that holds references to all of:

```
metadata, frames (briefly), timestamps, embeddings (briefly),
raw_deltas, smoothed, scores, changepoints | None,
peak_indices, peak_scores, padding_sec, budget_sec,
clips, merged, final_clips, clip_paths, reel_path
```

This is fine — `pipeline.run()` is the orchestrator. The rule is: **no module other than `pipeline.py` may hold more than its own inputs and outputs.** Specifically, `signal_processing.py` should never see `timestamps` (it works in index space); `clip_selection.py` is where index→seconds conversion happens.

---

## 4. The `--pelt` Branch — Where It Plugs In

The optional PELT path crosses two modules. Placement must be unambiguous.

### 4.1 Detection — in `signal_processing.py`

```python
# In signal_processing.py
def detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0) -> list[int]:
    import ruptures as rpt   # lazy import; --pelt-only dependency
    model = rpt.Pelt(model="rbf").fit(smoothed.reshape(-1, 1))
    return model.predict(pen=penalty)[:-1]
```

**Rationale:** PELT operates on the smoothed delta signal, which is `signal_processing.py`'s domain. `clip_selection.py` should never see the raw signal. Lazy import keeps `ruptures` out of the import graph when `--pelt` is off.

### 4.2 Application — in `clip_selection.py`

`apply_pelt_boost` lives here because it transforms peak scores (the input to clip building). The boost happens **after** `select_peaks` and **before** `build_clips`. JSON-time: `clip_selection.py` (or `pipeline.py` at JSON-assembly) sets `coincides_with_pelt_changepoint` per clip by checking `any(abs(idx - cp) <= 5 for cp in changepoints)`.

### 4.3 Orchestration — in `pipeline.py`

```python
# pipeline.py
if use_pelt:
    changepoints = detect_changepoints(smoothed)
else:
    changepoints = None

peak_indices, peak_scores = select_peaks(scores, height=height, min_gap_sec=min_gap_sec, fps=2.0)

if use_pelt:
    peak_indices, peak_scores = apply_pelt_boost(peak_indices, peak_scores, changepoints)
```

**JSON field rule:** when `use_pelt is False`, `coincides_with_pelt_changepoint = None` for all clips (per spec §8 + §5). When `True`, it's a bool computed per peak.

---

## 5. Data Flow — Annotated Walkthrough

For a 30-min video at 2 fps:

```
Step 1: extract.sample_frames(video_path, fps=2.0)
        → frames:     list[ndarray]           N ≈ 3600
        → timestamps: ndarray (3600,) float64

Step 2: extract.embed_frames(frames, model, preprocess)
        → embeddings: ndarray (3600, 768) float32, L2-normalized
        [del frames here]

Step 3: signal_processing.compute_deltas(embeddings)
        → raw_deltas: ndarray (3599,) float32, range [0, 2]

Step 4a: signal_processing.smooth_deltas(raw_deltas, kernel_size=5)
        → smoothed:   ndarray (3599,) float32

Step 4b: signal_processing.mad_normalize(smoothed, window_samples=180)
        → scores:     ndarray (3599,) float32, clipped to [0, 10]

Step 4c (--pelt only): signal_processing.detect_changepoints(smoothed, penalty=3.0)
        → changepoints: list[int], indices into signal

Step 5a: clip_selection.select_peaks(scores, height=1.5, min_gap_sec=15, fps=2.0)
        → peak_indices, peak_scores  (shape (P,) each, sorted by score desc)

Step 5b (--pelt only): clip_selection.apply_pelt_boost(...)
        → peak_indices, peak_scores re-sorted with 1.2× boost where coincident

Step 5c: budget_sec       = compute_budget_seconds(video_duration_sec)        # (dur/1800)*60
        padding_sec      = compute_padding(budget_sec)                       # max(3, min(8, budget*0.15))

Step 5d: clip_selection.build_clips(peak_indices, peak_scores, timestamps,
                                    video_duration_sec, padding_sec)
        → clips: list[(start, end, score)] chronological
        NB: This is where index→seconds conversion happens via timestamps[idx+1]
            (because scores[i] aligns to timestamps[i+1]).

Step 5e: clip_selection.merge_clips(clips, gap_threshold_sec=3.0)
        → merged

Step 5f: clip_selection.enforce_budget(merged, budget_sec)
        → final_clips

Step 6a: For clip in final_clips: export.extract_clip(...) → output/clips/{name}/clip_NNN.mp4
Step 6b: export.concat_clips(...) → output/reels/{name}_highlight.mp4
Step 6c: pipeline assembles JSON payload, utils.write_timestamps_json(...)
        → output/timestamps/{name}.json
```

### 5.1 The off-by-one — handle once, in `clip_selection.build_clips`

Because `compute_deltas` produces `N-1` values aligned to `timestamps[1:]`, the score at `scores[i]` corresponds to `timestamps[i+1]`. `build_clips` must use `peak_time = timestamps[idx + 1]`, not `timestamps[idx]`. This is the single most likely off-by-one bug in the pipeline. Lock it down with a unit test in the `signal_processing` / `clip_selection` boundary.

---

## 6. Build Order & Parallelism

The spec footer says: `extract.py → signal_processing.py → clip_selection.py → export.py → pipeline.py`. That's dependency-correct **if you serialize**. For parallel work across phases or contributors, the dependency graph is:

```
       extract.py
          │
          │ produces a fixture: (timestamps.npy, embeddings.npy)
          │
          ▼
   ┌──────┴──────────────────────────────────┐
   │                                         │
   │ signal_processing.py                    │ export.py
   │ (consumes embeddings, produces scores)  │ (consumes only video_path +
   │                                         │  hand-authored final_clips
   │                                         │  fixture)
   │                                         │
   │            ┌────────────────────────────┘
   ▼            ▼
   clip_selection.py ── (utils.py developed alongside, no blockers)
   (consumes scores + timestamps)
   │
   ▼
   pipeline.py
   (orchestration only — last)
```

### Recommended phase ordering (coarse-grained, parallel-aware)

| Phase | Module(s) | Depends on | Can run in parallel with |
|---|---|---|---|
| **P1** | `extract.py` + `utils.probe_video_metadata` | none (pure I/O + CLIP load) | nothing — produces the fixture everyone else needs |
| **P2a** | `signal_processing.py` | embeddings fixture from P1 | P2b, P2c |
| **P2b** | `clip_selection.py` (using a mocked `scores` fixture) | timestamps fixture from P1 + a hand-authored `scores.npy` | P2a, P2c |
| **P2c** | `export.py` (using hand-authored `final_clips`) | original video file, ffmpeg installed | P2a, P2b |
| **P3** | `pipeline.py` integration + JSON | all of P1, P2a, P2b, P2c | nothing |
| **P4** | `--pelt` branch (detect + boost + JSON field) | P3 working without PELT | P5 (parameter sweep) |
| **P5** | Run on remaining 4 videos, qualitative review | P3 working on video 1 | P4 |

**Key insight:** once P1 ships an `embeddings.npy` + `timestamps.npy` fixture from one sample video, P2a/P2b/P2c are independent. P2b and P2c can use synthetic inputs (a hand-authored scores array; a hand-authored 3-clip list) until P2a completes. **This is the parallelism the spec footer hides.**

**Critical serialization point:** P3 (pipeline glue) cannot start until **all** of P2a/b/c are done, because `run()` calls each in turn. There is no parallelism in P3.

**`--pelt` is deliberately deferred to P4.** Per the spec, the baseline must work without PELT first; PELT is opt-in supplementary. Starting it earlier risks tangling the JSON schema (the `coincides_with_pelt_changepoint = null` branch) before the non-PELT path is solid.

---

## 7. Output Artifacts — Who Writes What

| Artifact | Owner module | Triggered by | Notes |
|---|---|---|---|
| `output/reels/{video_name}_highlight.mp4` | `export.concat_clips` | step 6b | Final deliverable. |
| `output/clips/{video_name}/clip_NNN.mp4` | `export.extract_clip` | step 6a (×N) | Intermediate. Spec says "can be deleted after concat" — but **don't** delete by default; keep them for §0.5 verification ("Verify each intermediate clip file exists and is non-zero bytes"). Add a `--cleanup` flag later if disk space matters. |
| `output/timestamps/{video_name}.json` | `pipeline.run` (writer) + `utils.write_timestamps_json` | step 6c | The §8 schema; populated from values already in scope at the end of `run()`. |
| Concat manifest (`concat_manifest.txt`) | `export.concat_clips` | step 6b internal | Temp file; write to `output/clips/{video_name}/` and leave alongside the clips for debuggability. |
| Directory tree | `utils.ensure_output_dirs` | top of `run()` | Creates all three output subdirs idempotently with `os.makedirs(..., exist_ok=True)`. |

**Nothing else writes to disk in production.** §0.5 verification prints go to stdout, not files. If we later want diagnostic plots, that's an opt-in `--debug` flag and a `debug/` directory — out of scope for MVP.

---

## 8. Architectural Patterns Used

### Pattern 1: Linear pipeline with explicit fixtures

**What:** Each stage takes simple numpy/list inputs and returns simple numpy/list outputs. No shared state, no callbacks, no classes.

**When to use:** Single-process batch pipelines where the data flow is a known DAG.

**Trade-offs:**
- (+) Trivially testable — every function is `f(arrays) → arrays`.
- (+) Each stage's output is a serializable fixture (`np.save`/`np.load`), which directly enables the parallel build order in §6.
- (−) No streaming — the whole video's embeddings sit in RAM. For 30 min @ 2 fps × 768 dims × float32 = ~11 MB, this is fine. Would not scale to multi-hour input.

### Pattern 2: Lazy import for opt-in dependencies

**What:** `import ruptures` lives inside `detect_changepoints()`, not at module top.

**When to use:** Heavy or platform-fragile dependencies that some users won't exercise.

**Trade-offs:**
- (+) `--pelt`-off runs don't pay the import cost or fail if `ruptures` is misinstalled.
- (−) Slight inelegance; static import checkers won't flag the dependency.

### Pattern 3: Thin orchestrator + pure libraries

**What:** `pipeline.py` contains argparse, `run()`, and nothing else. All algorithmic decisions live in named functions in the 5 modules.

**When to use:** Anywhere you'd otherwise be tempted to put logic in `main()`.

**Trade-offs:**
- (+) Re-running pieces in a notebook for §0.5 verification is one import away.
- (+) Future "process all 5 videos in one process" wrapper just calls `run()` in a loop without re-loading the CLIP model — see §9.
- (−) Slightly more files to navigate.

### Pattern 4: Index-space vs time-space separation

**What:** `signal_processing.py` works entirely in sample-index space (no `timestamps` argument). `clip_selection.py` is the **single** place where `timestamps[idx+1]` converts indices to seconds.

**When to use:** Whenever signal processing and event localization are conceptually separate.

**Trade-offs:**
- (+) Eliminates a class of unit-confusion bugs ("is this seconds or samples?").
- (+) `signal_processing.py` is fully testable without ever constructing fake timestamps.

---

## 9. Performance & Scalability Notes

| Concern | At 30 min video | At 2 hr video | At 8 hr video (out of scope) |
|---|---|---|---|
| Frame count @ 2 fps | 3,600 | 14,400 | 57,600 |
| Embeddings RAM | ~11 MB | ~44 MB | ~177 MB |
| CLIP inference (CPU) | ~5–15 min | ~20–60 min | ~hours — would need GPU or temporal subsampling |
| MAD rolling window | O(N × window) ≈ 6.5e5 ops | 2.6e6 ops | 1e7 ops — still trivial |
| ffmpeg stream copy | seconds | seconds | seconds |

**The bottleneck is CLIP inference.** Everything downstream is sub-second. **Performance is not an architectural concern for this project** — the spec explicitly accepts CPU-only inference.

**One forward-looking note:** if multi-video runs become slow (5 videos × CPU inference), `pipeline.run()` should accept an already-loaded `(model, preprocess)` so the for-loop in the spec's `for f in videos/*.mp4; do python pipeline.py "$f"; done` can be replaced by an in-process loop that loads the model once. Architecturally, this means `extract.embed_frames` should accept `model` and `preprocess` as parameters (not load them itself), with a thin convenience wrapper that loads-and-embeds for single-shot use. **This is a 5-line change and worth doing now.**

---

## 10. Anti-Patterns to Avoid

### Anti-pattern 1: Putting algorithmic logic in `pipeline.py`

**What goes wrong:** Drifting toward a 300-line `run()` with merge logic, padding logic, and JSON assembly inline.
**Instead:** `run()` is a flat list of named function calls. If you find yourself writing a loop or a conditional with non-trivial body, it belongs in one of the 5 modules.

### Anti-pattern 2: Top-level `import ruptures`

**What goes wrong:** Users without `ruptures` installed can't even run the non-PELT pipeline. The spec explicitly says `ruptures` should not need to be imported when `--pelt` is off.
**Instead:** lazy import inside `detect_changepoints()`.

### Anti-pattern 3: Threading `timestamps` through `signal_processing.py`

**What goes wrong:** `signal_processing` becomes responsible for the index→seconds boundary, which fragments the off-by-one logic.
**Instead:** `signal_processing.py` is index-space only. Conversion happens once, in `clip_selection.build_clips`.

### Anti-pattern 4: Loading CLIP inside `embed_frames`

**What goes wrong:** Multi-video runs reload the model 5 times (~30s × 5).
**Instead:** Take `model, preprocess` as arguments; load once at the top of `pipeline.run()` (or in a future multi-video driver).

### Anti-pattern 5: Writing intermediate fixtures to disk in production

**What goes wrong:** `embeddings.npy`, `scores.npy` end up checked into the repo or filling `output/` with stale data.
**Instead:** Fixtures are a **development-time** convenience for parallel building (§6). Production runs hold everything in RAM. If you want a debug-mode dump, add `--debug` and write to `output/debug/{video_name}/`.

---

## 11. Confidence Notes

- **HIGH confidence** on module boundaries and data contracts: the spec is explicit and the data flow is mechanical.
- **HIGH confidence** on `--pelt` placement: the spec describes detection in §5 and application in §6 with code, mapping cleanly onto `signal_processing.py` and `clip_selection.py` respectively.
- **HIGH confidence** on the build order analysis: standard pipeline-with-fixtures pattern; the parallelism is real because no module other than `pipeline.py` shares mutable state.
- **MEDIUM confidence** on `utils.py` contents: the spec doesn't enumerate them; my recommendations (`probe_video_metadata`, `ensure_output_dirs`, `write_timestamps_json`) are derived from the call sites that need them, not from the spec.
- **MEDIUM confidence** on the timestamp alignment convention: the spec states it in §4 ("delta[i] corresponds to timestamps[i+1]") but does not put it in code. This is the single highest-risk off-by-one in the project; phase plans must include a unit test for it.

---

## 12. Sources

- `/Users/aryanahuja/abel_police_interview/assignment-details/bodycam_highlight_reel_spec.md` §§1, 3–9, 0.5, 12 — the locked design document, treated as the authoritative source.
- `/Users/aryanahuja/abel_police_interview/.planning/PROJECT.md` — confirms the locked module structure and the build-order footer.
- No external sources required for this dimension — the architectural shape is fully determined by the spec; this document validates and tightens it.
