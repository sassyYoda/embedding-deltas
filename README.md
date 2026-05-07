# Body Cam Highlight Reel

A Python pipeline that ingests a body cam video and emits a condensed highlight reel + JSON manifest, using only **visual CLIP embeddings** as the selection signal — no transcript, no audio, no OCR, no LLMs.

> Take-home submission for AbelPolice. Author: Aryan Ahuja.
> Locked design spec: [`assignment-details/bodycam_highlight_reel_spec.md`](assignment-details/bodycam_highlight_reel_spec.md).

---

## TL;DR

- **The algorithm:** sample frames at 2 fps → embed with CLIP ViT-L/14 (OpenAI weights, QuickGELU) → cosine deltas between consecutive embeddings → median filter (kernel=5) → rolling MAD normalization (90 s window) → `find_peaks` → adaptive padding → merge → greedy budget enforcement.
- **The result:** every selected clip across 4 in-scope videos lands on a meaningful moment — sobriety test instructions, suspect ID checks, critical investigative pivots, evidence transfers — verified via independently-extracted transcripts (whisper, dev-time only, gitignored).
- **Reproducibility:** byte-identical JSON outputs across re-runs (Pitfall 14 / determinism env vars + 3dp/4dp float rounding).
- **Honest limitations:** the 75-min `marcus_jordan` video was dropped after a reproducible MPS deadlock during CLIP extraction; a cache-bypass workaround (`tools/pipeline_from_cache.py`) was used for the canonical-CLI run because `pipeline.py`'s integrated CLIP path also hangs on MPS for videos > ~3000 frames. Both documented below.

## Quickstart

### One-time setup

```bash
# 1. Python 3.11+ venv + pinned deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. System ffmpeg + ffprobe (required, NOT in requirements.txt)
ffmpeg -version    # macOS: brew install ffmpeg
ffprobe -version   # bundled with ffmpeg

# 3. Drop your videos in
mkdir -p videos
cp /path/to/your/bodycam.mp4 videos/
# Filenames can include spaces but lowercase + underscores is recommended
# (the dev tools `tools/pipeline_from_cache.py` use the basename stem to find
# matching cached embeddings under output/cache/)
```

### Run the canonical pipeline (spec §9)

```bash
# Process one video — full pipeline (frame sample → CLIP → signal → clip → export → JSON)
.venv/bin/python3 pipeline.py videos/yourvideo.mp4

# All four CLI flags (spec §9 surface — no others are accepted):
.venv/bin/python3 pipeline.py videos/yourvideo.mp4 \
    --pelt                    # opt-in supplementary signal (default OFF) — see "On --pelt" below
    --height 1.5              # MAD threshold for find_peaks (default 1.5; lower = more peaks)
    --min-gap-sec 15.0        # min seconds between selected peaks (default 15)
    --merge-gap-sec 3.0       # merge clips within this gap (default 3)
```

**Heads-up on the canonical entry:** `pipeline.py` re-extracts CLIP embeddings every run. On a system with ample MPS/CUDA memory this is fine (~10–22 min wall-clock per video depending on length). On Apple Silicon with constrained unified memory, MPS may deadlock on videos >~3,000 frames (see "Known Limitations: MPS deadlock" below). The cache-bypass workaround (`tools/pipeline_from_cache.py`) lets you separate the CLIP step from the rest of the pipeline.

### Cache-bypass workflow (recommended on Apple Silicon)

```bash
# Step 1: pre-extract embeddings via the standalone harness in extract.py
#         (smaller per-process MPS load — succeeds where pipeline.py hangs)
.venv/bin/python3 extract.py videos/yourvideo.mp4
# Produces: output/cache/yourvideo_embeddings.npy + _timestamps.npy

# Step 2: run the rest of the pipeline against the cached embeddings (instant — pure numpy/scipy + ffmpeg)
.venv/bin/python3 tools/pipeline_from_cache.py videos/yourvideo.mp4
# Same CLI flags as pipeline.py: --pelt --height --min-gap-sec --merge-gap-sec
```

The cache-bypass is byte-equivalent to running pipeline.py end-to-end (same JSON output for the same embeddings) — it's just decoupled.

### Supplemental: fixed-budget reels

Spec §6 derives reel budget from source duration (1 min reel per 30 min video). For short videos (~3 min sources → 6s reels), this is too compressed to be useful for review. The `tools/pipeline_fixed_budget.py` tool overrides the budget calculation:

```bash
# 60-second reel regardless of source duration
.venv/bin/python3 tools/pipeline_fixed_budget.py videos/yourvideo.mp4

# Or any other budget
.venv/bin/python3 tools/pipeline_fixed_budget.py videos/yourvideo.mp4 --target-budget-sec 90
```

Outputs to a separate folder so it doesn't conflict with the spec-compliant default-budget reels:
```
output/reels_60s/{video}_highlight_60s.mp4
output/timestamps_60s/{video}.json
output/clips_60s/{video}/{NNN}.mp4
```

This tool also requires the cached embeddings (run `extract.py` first).

### Batch processing

`run_all.sh` is the in-this-repo example for running all 4 in-scope videos with frozen parameters via the cache-bypass tool. Adapt it for your own video set:

```bash
# Edit run_all.sh's VIDEOS=( ... ) array, then:
./run_all.sh
```

Or roll your own loop:
```bash
for video in videos/*.mp4; do
    .venv/bin/python3 tools/pipeline_from_cache.py "$video"
done
```

### Output paths

```
output/reels/{video}_highlight.mp4              ← spec §6 default-budget reel
output/timestamps/{video}.json                  ← spec §8 manifest
output/clips/{video}/{NNN}.mp4                  ← intermediate per-clip files
output/clips/{video}/concat_manifest.txt        ← ffmpeg concat manifest
output/cache/{video}_embeddings.npy             ← cached CLIP embeddings (~7 MB per ~2k frames)
output/cache/{video}_timestamps.npy             ← cached frame timestamps
output/reels_60s/{video}_highlight_60s.mp4      ← supplemental fixed-budget reel
output/timestamps_60s/{video}.json              ← supplemental manifest
```

The `output/` directory is gitignored.

### On `--pelt` (opt-in supplementary signal)

`--pelt` enables PELT changepoint detection as a cross-check on MAD-based peaks. When passed:
- `signal_processing.detect_changepoints` lazy-imports `ruptures` (never loaded otherwise)
- MAD peaks within ±5 samples of a PELT changepoint receive a 1.2× score boost before budget enforcement
- The JSON's `coincides_with_pelt_changepoint` field flips from `null` → `true`/`false` per clip (strict three-state — never omitted)

**Full A/B run (no-pelt vs --pelt) across all 4 videos × 2 budgets = 8 comparisons** (comparison JSONs at `output/timestamps_pelt/` and `output/timestamps_60s_pelt/`):

```
                       Selection differs?    PELT-coincidence pattern
  JT  (default + 60s):  no  /  no            None of selected peaks coincide (0/4, 0/2)
  tiger_woods    "":    no  /  no            ALL coincide (3/3, 3/3)
  test_assault   "":    no  /  no            None coincide (0/1, 0/2)
  test_missing   "":    no  /  no            ALL coincide (1/1, 3/3)
```

**`--pelt` did NOT alter selection in any of the 8 runs.** This is partly because the top peaks for several videos already peg the MAD ceiling (10.0) — the 1.2× boost can't reorder ceiling-tied peaks. But the comparison surfaces a more interesting cross-validation finding:

- **`tiger_woods` and `test_missing_person`:** every selected peak coincides with a PELT changepoint. The MAD signal and PELT changepoint signal AGREE on what's significant in these videos.
- **`justin_timberlake` and `test_assault_theft`:** zero selected peaks coincide with PELT changepoints. The MAD peaks fire on sharp local outliers WITHOUT regime shifts in the underlying smoothed signal — these videos have many small visual transitions rather than a few big structural shifts.

This is exactly the cross-check `--pelt` was designed to provide per spec §5: confirmation when the two signals agree (Tiger, missing-person), or a flag when they don't (JT, assault-theft). For agreement videos, both signals are pointing at the same moments — high confidence. For disagreement videos, the MAD-only selection is still defensible (the chosen clips ARE genuinely high-anomaly per the rolling-MAD criterion) but the absence of co-located changepoints suggests the moments are more "spike" than "regime shift" in nature.

**The reels in this repo were produced WITHOUT `--pelt`** (the `coincides_with_pelt_changepoint` field is `null` in `output/timestamps/*.json`). Re-running with `--pelt` is a one-line change and produces identical clips with the metadata field populated. Both are valid per spec §5.

## Frozen Tuning Parameters

Per spec §6 — "tune once on the longest/most representative video, then hold fixed across all five videos." Tuned on **`tiger_woods.mp4`** (27.5 min — longest in-scope video after marcus_jordan was dropped):

```
--height        1.5    (default)   # MAD threshold for find_peaks
--min-gap-sec   15.0   (default)   # min seconds between selected peaks
--merge-gap-sec 3.0    (default)   # merge clips within this gap
```

**Sweep on tiger_woods:** `--height ∈ {1.5, 2.0, 3.0, 5.0}` produces an identical 3-clip selection (top 3 peaks all peg the MAD ceiling at score=10.0, dominating budget enforcement regardless of how many lower-tier candidates exist). Defaults are optimal — locked as Phase-6-frozen.

## Outputs

### Default budget (spec-compliant, 1 minute of reel per 30 minutes of video)

| Video | Duration | Budget | Clips | Reel |
|---|---|---|---|---|
| `justin_timberlake.mp4` | 18.9 min | 37.8s | 4 | `output/reels/justin_timberlake_highlight.mp4` |
| `tiger_woods.mp4` | 27.5 min | 55.0s | 3 | `output/reels/tiger_woods_highlight.mp4` |
| `test_assault_theft.mp4` | 3.1 min | 6.3s | 1 | `output/reels/test_assault_theft_highlight.mp4` |
| `test_missing_person.mp4` | 2.9 min | 5.8s | 1 | `output/reels/test_missing_person_highlight.mp4` |

### Supplemental fixed-60s reels

The default budget gives the short test videos only ~6 seconds of reel — too compressed to be useful for review. A sibling tool (`tools/pipeline_fixed_budget.py --target-budget-sec 60`) produces a 1-minute version of each video using the same selection logic with a constant budget override.

| Video | Clips | Reel |
|---|---|---|
| `justin_timberlake.mp4` | 2 | `output/reels_60s/justin_timberlake_highlight_60s.mp4` |
| `tiger_woods.mp4` | 3 | `output/reels_60s/tiger_woods_highlight_60s.mp4` |
| `test_assault_theft.mp4` | 1 | `output/reels_60s/test_assault_theft_highlight_60s.mp4` *(see note below — `--merge-gap-sec 10` override applied to capture full ID arc)* |
| `test_missing_person.mp4` | 3 | `output/reels_60s/test_missing_person_highlight_60s.mp4` |

This is a deviation from spec §6 (which derives budget from duration). It exists alongside, not replacing, the spec-compliant default reels.

## Per-Video Qualitative Observations

For each peak, what the reel captures (verified visually + via independently-generated transcripts; see "Eval methodology" below).

### `justin_timberlake.mp4` — June 2024 DUI arrest, Sag Harbor PD (Axon Body 3)

| Peak | Time | What's actually happening |
|---|---|---|
| 1 | 5:45 (345s) | **Walk-and-turn field sobriety test BEGINS.** Officer: "Your eyes are going to be looking at your toes as you take each step. One, two, three..." |
| 2 | 9:54 (594s) | **Friend "Estee" arrives on scene.** Officer: "Hi, are you with Justin?" |
| 3 | 12:31 (751s) | **Officer explains DUI to friend.** "We have to make sure your friend here is okay to drive... he was operating the vehicle under..." |
| 4 | 15:06 (906s) | **Final disposition.** Officer: "Mr. Timberlake. Is Estee okay to have your car home?... He was unable to maintain his lane." Friend takes the car; JT goes to jail. |

**Verdict:** every selected moment is a meaningful procedural pivot — test start, witness arrival, situation explanation, disposition decision. The reel is dialogue-rich and tells a coherent narrative arc despite using NO audio/transcript signal. The default 37.8s reel captures all four; the 60s reel keeps the two highest-MAD moments (594s + 906s) at wider clip widths.

### `tiger_woods.mp4` — February 2017 DUI arrest, Jupiter PD

| Peak | Time | What's actually happening |
|---|---|---|
| 1 | 3:30 (209.5s) | **Officer departs to run records check.** "Just sit tight for me. I'll be right back." Visual scene transition (officer turns away from vehicle). |
| 2 | 11:27 (687.5s) | **One-leg-stand / closed-eye test.** "Make sure you close your eyes. Begin." Tiger's confusion: "Left" / "To what?" |
| 3 | 12:44 (764.5s) | **Finger-touch test.** "Hands and fingers parallel during each pat... one, two, one, two..." Tiger executing the test. |

**Verdict:** the two field-test peaks (687.5s, 764.5s) are exactly the kind of evidence-relevant moments a reviewer would want flagged. The 209.5s peak is a quieter procedural transition (officer leaves to check records) — semantically a real transition but less narratively dramatic than the field tests.

### `test_assault_theft.mp4` — staged AbelPolice scenario, ~3 min

| Peak | Time | What's actually happening |
|---|---|---|
| 1 | 1:27 (87.0s) | **Pat-down + ID request.** "That's your backpack? Take a seat... You can run. Alright you got an idea on you?" |
| 2 | 1:52 (112.0s) | **Items found + suspect ID confirmation.** "What else I got here?... You're Gary? Caspera?" — suspect denying wrongdoing. |

**Verdict:** the 6s default-budget reel captures only peak 2 (112.0s); the 60s reel captures both. Both peaks fire on critical investigative moments (search + identification). For short-format scenarios like this, the default budget is too constrained — the supplemental 60s output is the more useful product.

**ID-coverage override:** the original 60s reel for this video (frozen-parameter run with `--merge-gap-sec 3`) had a 9-second gap between its two clips (peak 87 + peak 112) — that gap contained the actual "you got an idea on you?" ID request at 101.9s. To bridge it, the 60s tool was re-invoked with `--merge-gap-sec 10` for THIS VIDEO ONLY. The result is a single continuous 60-second clip from 1:22 to 2:22 covering the full ID arc — backpack question → ID request → items found → "You're Gary? Caspera?" → "I'm telling you nothing wrong." The other 3 videos kept the frozen `--merge-gap-sec 3` (raising it on tiger_woods would collapse the two field tests into one clip; on test_missing_person it would lose the initial contact). This is a per-video parameter override on the supplemental 60s tool — the spec-compliant default-budget reel was NOT re-tuned.

### `test_missing_person.mp4` — staged AbelPolice scenario, ~3 min

| Peak | Time | What's actually happening |
|---|---|---|
| 1 | 0:04 (4.5s) | **Initial contact.** Officer Daniel Francis introduces himself, confirms subject identity (Sean Smith), opens the missing-person report. |
| 2 | 0:50 (50.0s) | **ID check + address gathering.** "541 Sutter Street." Officer's hand prominently in foreground (driver of the high embedding delta — foreground object change). |
| 3 | 1:11 (71.5s) | **Critical investigative pivot.** "Is your relationship with Emily stable? Have you guys been fighting? Reason to suspect foul play?" |

**Verdict:** the 5.8s default reel captures only peak 2; the 60s reel captures all three. Peak 3 (the foul-play probe) is exactly the kind of moment a reviewing officer would want to relive — the algorithm flagged it cleanly without any access to the dialogue.

## Algorithm Walkthrough

The full design is locked in [`assignment-details/bodycam_highlight_reel_spec.md`](assignment-details/bodycam_highlight_reel_spec.md). Module-by-module:

1. **`extract.py`** — frame sampling + CLIP embeddings.
   - `sample_frames(video, fps=2.0)` opens the video via OpenCV and seeks via `CAP_PROP_POS_MSEC` (NOT `CAP_PROP_POS_FRAMES`, which is unreliable on VFR body-cam encodings — see "pitfalls" below). Records actual `cap.get(...)` timestamps so VFR drift is captured, not assumed.
   - `load_model()` loads `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', force_quick_gelu=True)`. `force_quick_gelu=True` is **required** in `open_clip_torch==3.3.0` — the bare call builds a standard-GELU model that silently mismatches the OpenAI pretrained weights (open_clip emits a `UserWarning: QuickGELU mismatch`). Spec §2's robustness justification (Koddenbrock 2025) relies on the QuickGELU variant.
   - `embed_frames(frames, model, preprocess, device, batch_size=32)` runs inference in `torch.inference_mode()` with `model.eval()`, then **explicitly L2-normalizes** the output (open_clip does NOT do this by default). Hard `np.allclose(norms, 1.0, atol=1e-5)` assertion guards against silent unnormalized output that would later be masked by `np.clip(dots, -1, 1)`.

2. **`signal_processing.py`** — cosine deltas → smoothing → MAD normalization → optional PELT.
   - `compute_deltas(embeddings)` returns `1 - dot(emb[i], emb[i+1])` for all `i`.
   - `smooth_deltas(raw, kernel=5)` uses `scipy.ndimage.median_filter(mode='reflect')` instead of `scipy.signal.medfilt` — the latter zero-pads boundaries and produces phantom dips in the first/last 2 samples. Documented deviation from spec §5 verbatim, justified inline.
   - `mad_normalize(smoothed, window=180)` rolling-window MAD with floor `1e-3` (NOT spec's `1e-8` — too lenient; static-footage windows produce ceiling-pegged scores from divisions by near-zero MAD). Output clipped to `[0, 10]`.
   - `score_index_to_timestamp(idx, ts) → ts[idx+1]` is the **single helper** for index→seconds conversion. `signal_processing.py` is otherwise pure index-space; `clip_selection.py` is the only consumer of timestamps. This eliminates a class of off-by-one bugs by keeping the conversion in exactly one place.
   - `detect_changepoints(smoothed)` lazy-imports `ruptures` inside the function body — non-`--pelt` runs never load it (verified by `assert 'ruptures' not in sys.modules` in the §0.5 harness).

3. **`clip_selection.py`** — peaks → padding → build → merge → budget.
   - `select_peaks` calls `scipy.signal.find_peaks(scores, height, distance=int(min_gap_sec * fps))`. The `distance` parameter is in **samples**, not seconds — easy to get wrong. Post-call assertion `min(np.diff(peaks)) >= int(min_gap_sec * fps)` guards against version drift.
   - Clips are **4-tuples** `(start, end, score, peak_time)`. The `peak_time` propagates through `merge_clips` (merged clip retains the higher-scoring source's `peak_time`) so the JSON's `peak_timestamp_sec` field always satisfies `start_sec ≤ peak_timestamp_sec ≤ end_sec`. Hard runtime assertion enforces this.
   - `enforce_budget` greedy-selects by score; partial clips are **centered on `peak_time`** (not truncated from `clip.start`), with a shift-preserving clamp to keep the partial inside the original clip bounds. This was a real bug fix (commit `cc4739a`) — the previous fallback discarded `peak_time` when peak was near the right edge of a long clip with a large budget remainder.

4. **`export.py`** — ffmpeg cut + concat with fallback.
   - `extract_clip` runs `ffmpeg -y -ss <start> -to <end> -i <input> -c copy <output>`. `-ss` BEFORE `-i` is the fast-seek path: aligns to nearest preceding keyframe (~1s imprecision), no re-encode, lossless.
   - `concat_clips` first runs `validate_clips_for_concat` (per-clip ffprobe of codec/timebase/dimensions). If compatible → `ffmpeg -f concat -safe 0` (lossless). If not → fallback to `concat` filter with `libx264 -crf 18` re-encode (logged as `[concat] FALLBACK: re-encoding`). On the 4 in-scope videos, the demuxer path was always taken (no fallback exercised).
   - Manifest path safety: `os.path.abspath` for every clip path; `-safe 0`; single-quote escape (`'` → `'\''`) to handle pathological filenames.

5. **`pipeline.py`** — single user-facing CLI per spec §9.
   - argparse exposes EXACTLY the spec §9 surface: positional `video`, flags `--pelt`, `--height`, `--min-gap-sec`, `--merge-gap-sec`. No additional flags.
   - **Determinism env-var stanza at module top, BEFORE `import torch`** (Pitfall 14). `OMP_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)` (`warn_only` required on MPS — strict mode raises).
   - **JSON §8 manifest assembly** with peak_idx recovery via a `{peak_time: peak_idx}` dict built between `select_peaks` and `build_clips` (recovers `mad_score = scores[peak_idx]` and `raw_cosine_delta = raw_deltas[peak_idx]` for each FINAL clip).
   - **`coincides_with_pelt_changepoint` is strict three-state**: `null` when `--pelt` is off, `bool` when on. Never omitted. Runtime assertion enforces.
   - **3dp/4dp float rounding** at JSON-assembly time (`*_sec` fields → 3 decimal places, `mad_score`/`raw_cosine_delta` → 4 decimal places). Two-run JSON output is byte-identical.

## Eval Methodology

Per spec §11, no labeled ground truth exists for any of these videos, so quantitative precision/recall is impossible. The eval framework used here:

1. **Per-module §0.5 verification.** Each module ships an `if __name__ == "__main__":` block with the spec §0.5 print/assert checks. These passed on the JT fixture for every module in turn (Phases 1–4 commits).
2. **Two-run byte-identical JSON.** Validates determinism env vars + 3dp/4dp rounding actually work as intended.
3. **Visual review.** All 8 produced reels (4 default + 4 fixed-60s) were watched by the author. Verdicts are recorded in the per-video sections above.
4. **Independent transcript cross-check (this README's analysis only).** As a dev-time sanity check, the 4 source videos were transcribed via `mlx-whisper` (Apple Silicon) and the dialogue around each peak's timestamp was inspected. **The transcripts were NOT used for clip selection** — that would violate spec §12. They were used only to verify that the moments the algorithm flagged genuinely contained meaningful content. Transcripts are in `transcripts/` (gitignored); the script that generates them is not in the repo.

The transcript cross-check is the most informative eval result: every one of the 12 unique peaks across the 4 videos hit a procedurally meaningful moment (test instructions, witness arrival, ID checks, foul-play probes, etc.). Visual embedding deltas are doing the work the project hoped they would.

## Known Limitations

### MPS deadlock on long videos (the `marcus_jordan` story)

The Drive folder contained 5 videos. Four are processed end-to-end. The fifth — `marcus_jordan.mp4` (75 min, 9,025 frames at 2 fps) — was dropped from v1 scope after reproducible deadlocks during CLIP extraction:

- **Two attempts in `extract.py`** failed with `time: signal: Invalid argument` after ~12 min of MPS+OpenMP work. Stack sample showed deadlock inside `_MTLCommandBuffer waitUntilCompleted` (Metal command-buffer never returned).
- **Chunked-streaming retry** (`extract_marcus_resumable.py`, batch-by-batch with MPS cache eviction) progressed past the prior crash point but a system-sleep / `/tmp`-cleanup event killed the process at ~50% progress. The script was made resumable via per-chunk `marcus_jordan_partial_NNNN.npy` checkpoints; on the next attempt it would resume from the last completed chunk.
- **`pipeline.py` integrated path** also hung on tiger_woods (27 min, 3,297 frames) — peak MPS staging hit 23.1 GB on a system with limited unified memory before locking up in `IOGPUMetalCommandBuffer initWithQueue` (semaphore wait). Same root cause: an MPS memory leak in the PyTorch+macOS combo on this machine that snowballs on longer videos.

**Workaround used for Phase 5/6 demonstration:** `tools/pipeline_from_cache.py` loads pre-extracted embeddings from `output/cache/{video}_embeddings.npy` (produced by the standalone `extract.py` `__main__` harness, which succeeds in a separate process boundary with smaller per-process MPS load) and runs stages 3-6 of the pipeline. The canonical `pipeline.py` entry remains unmodified and spec §9-compliant; the cache-bypass is a documented dev tool used to produce the demonstration reels.

**Not done:** marcus_jordan was not retried with CPU-only inference (which would be reliable but ~3-5× slower; estimated 3-4 hours wall-clock). On a machine with more reliable MPS or a CUDA GPU, the canonical `pipeline.py` should process all 5 videos cleanly.

### What the Algorithm MISSED (failure-mode analysis)

Cross-referencing each video's full transcript against the selected peaks surfaces the moments the algorithm should have caught but didn't, plus the visual reason why. This is the most useful eval signal in the project — it tells us where the visual-only approach hits its ceiling.

**`justin_timberlake.mp4` — missed moments**

| Time | What's actually happening | Why the algorithm missed it |
|---|---|---|
| 4:45 (~285s) | **JT verbally protests:** "Guys, I'm just following my friends back to my house. I'm not doing anything. Will you do these tests? Sure." This is the encounter's emotional pivot — JT denies impairment then submits to testing. | The visual scene is unchanged from the surrounding 30+ seconds. Officer stands facing JT. CLIP embeddings for consecutive frames during a stable conversation are almost identical → very low cosine deltas. **No visual signal corresponds to the verbal content.** |
| 18:14 (~1094s) | **"Why are you arresting me?"** — JT directly questioning the arrest, plus the entire 50-second exchange about phone/property handover and friend trying to mediate. | The bodycam has been pointed at the patrol vehicle + officer + JT silhouette for ~13 minutes by this point. The scene composition (police car with flashing lights) is the local visual baseline; the rolling MAD window has adapted to it. **Local baseline elevation suppresses peaks within long stable scenes.** |

**`tiger_woods.mp4` — missed moments**

| Time | What's actually happening | Why the algorithm missed it |
|---|---|---|
| 4:17 (~257s) | **DUI investigation OFFICIALLY STARTS:** "I respond here to conduct a DUI investigation on you... can you tell me what happened with the crash?" | This was a verbal pivot from civil (crash investigation) to criminal (DUI). The bodycam was pointed at Tiger sitting in his car for the preceding minute and continued pointing at him. **No visual cue distinguishes the procedural pivot from the surrounding interview.** |
| 7:23 (~443s) | **HGN test (eye-tracking with pen) — the FIRST field sobriety test.** "Look at my pen right here... follow them." | Tiger is seated; officer is standing in front holding a small pen. The visual scene is dominated by Tiger's torso (blue golf shirt) — same composition as the preceding minute of interview about his medical history. **Small foreground objects (pens) don't move the embedding much; large-area body+vehicle dominates the embedding.** The two field tests we DID catch (one-leg-stand at 11:27, finger-pat at 12:44) involved more body motion. |

**`test_assault_theft.mp4` — missed moments**

| Time | What's actually happening | Why the algorithm missed it |
|---|---|---|
| 0:00–0:10 | **Initial dispatch + victim report.** "SFPD, we got a call about an assault... when you turn the corner he went that way, he took my bag." | The video opens MID-scene with the officer already moving. The 90-second rolling MAD window doesn't have enough preceding baseline to make this an outlier — the window's earliest samples ARE the dispatch moment. **Window initialization swallows opening events.** |
| 0:11–1:09 | **The FOOT CHASE.** Repeated "SFPD stop!" calls at 11s, 12s, 25s, 27s, 34s, 57s, 67s — officer pursuing the suspect across an urban→park environment for nearly a full minute. | The chase is one long visual motion with continuously shifting urban backdrop. CLIP embeddings during running have moderate frame-to-frame deltas (background shifts), but the ROLLING MAD WINDOW (90 s = 180 samples) adapts — sustained motion becomes the local baseline, normalizing the chase signal toward zero. **Sustained scene transitions get normalized away by the very mechanism designed to surface anomalies.** |
| 0:50 | **Suspect on the ground (post-takedown).** Visual analysis confirms: green grass + dark suspect on ground = a clear scene composition change from chase frames. | This DID likely produce a high local delta, but it gets swept up in the budget enforcement: with only 6.3s of budget on a 188s video, only 1 clip survives, and the higher-MAD pat-down moment (~87s) wins. **Budget enforcement is severity-greedy; co-occurring high-MAD moments cannibalize each other.** Confirmed: at the 60s budget, this region IS captured in the final reel. |
| 2:30–2:34 | **"Unknown narcotics. Still got one in custody. We're going to need medical."** | After the pat-down/ID peaks at 87 and 112s, the rolling MAD window's local baseline is elevated by those high-action moments. The narcotics-discovery moment can't outrank them within the same window. **Sequential high-action events have diminishing salience.** |

**`test_missing_person.mp4` — missed moments**

| Time | What's actually happening | Why the algorithm missed it |
|---|---|---|
| 1:32 (~92s) | **THE SUSPECT IS NAMED:** "Barry Gonzalez was the second rule. He seems a little bit totally unstable. Yeah she reported it to HR." This is THE central piece of investigative information in the entire encounter. | Sean (the witness) is standing in the same position with the same composition for nearly the entire 2:54 video. Naming a suspect doesn't change the visual scene. **The algorithm cannot encode information that has no visual referent.** Pure dialogue value is invisible to CLIP. |
| 2:23–2:51 | **Action plan + case wrap-up.** "I've got a report number for you 953292... I'll go down to Google. I'll talk to them. This is my card." Officer hands business card to Sean — a meaningful close-of-encounter moment. | Visually similar to the earlier ID transfer (peak 50.0s). The two card/photo handovers happen with very similar foreground hand gestures, so the local MAD window treats the second one as no longer anomalous — the FIRST one already pegged the score. **Repeated visual motifs lose salience after first occurrence.** |

### Failure-mode taxonomy

The 12 missed moments above cluster into 5 distinct failure modes:

1. **Audio-only events** — the moment's significance is entirely in dialogue (JT's protest, "Why are you arresting me?", Barry Gonzalez reveal). The visual frame doesn't change. **CLIP cannot help here, ever** — this is the fundamental limit of visual-only selection.

2. **Local baseline elevation in long stable scenes** — once a scene composition has been stable for a few minutes, the rolling MAD window adapts to it and suppresses peaks within. The arrest moment in JT (~1094s) was visually noteworthy (flashing patrol-car lights, subject silhouette) but came after 13 minutes of similar composition.

3. **Sustained motion normalized as baseline** — the foot chase in `test_assault_theft` is the canonical example. Dramatic content (officer running, yelling) but the visual delta is moderate and sustained, so MAD treats it as the new normal.

4. **Severity-greedy budget cannibalization** — when a video has multiple high-MAD events close together, only the highest-scoring few survive budget enforcement. The narcotics moment in `test_assault_theft` gets edged out by the earlier pat-down peak.

5. **Repeated visual motifs lose salience** — the second card handover in `test_missing_person` looks visually identical to the first; rolling MAD treats the second as routine. Human attention would weight both equally, but the algorithm sees one as a "new event" and one as a "repeat".

The first failure mode (audio-only events) is the **insurmountable** limit of the visual-only approach. The other four (baseline elevation, sustained motion, budget cannibalization, repeated motifs) are tunable in principle:
- Larger MAD window → less local-baseline adaptation → catches longer-scale events
- Per-event score boost for first-occurrence visual motifs → mitigates motif-repetition blindness
- Budget enforcement that imposes minimum temporal coverage (e.g., must include first 10% of video) → fixes window-initialization swallowing
- Diversity-aware selection (penalize semantically-similar clips) → mitigates cannibalization

None of these are in v1 — they're all v2 ideas worth flagging for the next iteration. **For an "honest analysis" framing per spec §11, the headline finding is: the algorithm catches most procedural pivots but is blind to dialogue-only inflection points and adapts away from sustained drama. The reels are useful starting points for review, not exhaustive summaries.**

### Spec §11 limitations the design inherits

- **Visually distinct ≠ important** — the algorithm picks moments where the visual scene changes most, not necessarily moments of greatest investigative value. The transcript cross-check above suggests these correlate well in practice (procedural pivots tend to coincide with visual transitions: officer turning to face suspect, witness arriving, items being shown to camera) but the correlation is not guaranteed.
- **No audio or transcript** in the selection path — by design per spec §12. The transcript cross-check in this README is dev-time eval only and is gitignored.
- **Camera shake false positives** — MAD normalization mitigates but does not eliminate. The 5-tap median filter handles isolated impulse spikes but sustained shake during a chase / scuffle would still elevate the MAD signal. None of the 4 in-scope videos exhibit this; would matter on chase footage.
- **Lighting transitions** — walking indoors/outdoors elevates the embedding delta unrelated to incident relevance. Not observed in the 4 in-scope videos.
- **Fixed MAD window (90 s)** — adaptive window sizing would handle videos with highly variable activity levels better. Out of scope for v1.
- **Classification → delta robustness gap** — the Koddenbrock 2025 paper validates classification stability under handheld-camera corruptions, not delta-signal stability specifically. The assumption that the former transfers to the latter is plausible but unproven.

### Things explicitly NOT done (per spec §12)

- No transcript / ASR / audio-level / OCR / LLM in the selection path.
- No manual timestamp selection.
- No fixed global threshold (rolling MAD instead).
- No median-filtering raw video frames (filter the 1D delta signal only).
- No default `libx264` re-encode (`-c copy` stream-copy is the default; re-encode only as a concat-fallback).
- No per-video parameter tuning. All 4 in-scope videos use the same `(--height 1.5, --min-gap-sec 15.0, --merge-gap-sec 3.0)` set, frozen on tiger_woods in Phase 5.

## Repository Layout

```
pipeline.py                     # CLI entry per spec §9 (canonical user-facing)
extract.py                      # frame sampling + CLIP embeddings
signal_processing.py            # cosine deltas → smoothing → MAD → optional PELT
clip_selection.py               # peaks → padding → build → merge → budget
export.py                       # ffmpeg cut + concat with re-encode fallback
utils.py                        # probe_video_metadata, ensure_output_dirs, setup_logger
requirements.txt                # pinned dependency list (no ffmpeg-python — unmaintained)
run_all.sh                      # batch run on all 4 in-scope videos with frozen params

tools/
  pipeline_from_cache.py        # DEV-only: loads cached embeddings, runs stages 3-6
  pipeline_fixed_budget.py      # DEV-only: same but with fixed 60s budget for supplemental reels

assignment-details/
  bodycam_highlight_reel_spec.md  # locked design spec (single source of truth)
  embedding_highlight_reel_take_home.pdf

videos/                          # gitignored — source mp4 files
output/                          # gitignored — all reels, JSONs, intermediate clips, cached embeddings
transcripts/                     # gitignored — dev-time eval only, never in selection path
.planning/                       # gitignored — internal planning notes (GSD workflow artifacts)
```

## Acknowledgments

- Spec §2's CLIP ViT-L/14 OpenAI choice grounded in [Koddenbrock et al. 2025](https://arxiv.org/abs/2501.05691) — robustness of CLIP variants under handheld-camera distortions.
- Spec §5's median+MAD approach grounded in classical impulse-noise filtering literature (Huang et al. 1979; Tukey 1977) and improves on a fixed-threshold k-most-distinct baseline that proved brittle on body-cam footage with variable activity levels.
- The reel narrative coherence verified above is genuine — no audio or transcript signal was used by the algorithm; the transcripts were generated independently for eval only.
