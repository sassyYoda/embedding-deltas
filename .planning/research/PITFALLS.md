# Pitfalls Research

**Domain:** Visual-embedding-based highlight extraction (CLIP → 1D signal → ffmpeg) on body cam footage
**Researched:** 2026-05-06
**Confidence:** HIGH (drawn from documented behaviors of OpenCV, open_clip, scipy, and ffmpeg; complements but does not duplicate spec §11/§12)

> **Scope note:** This document catalogs *implementation* pitfalls — the seams where a correct-on-paper pipeline produces wrong outputs. Domain-level limitations (visually distinct ≠ important, lighting/shake confounders, classification→delta gap) are covered in spec §11 and not repeated here.

---

## Critical Pitfalls

### Pitfall 1: BGR→RGB conversion missing or applied twice

**What goes wrong:**
OpenCV `VideoCapture.read()` returns frames in **BGR** order. CLIP's `preprocess` transform expects **RGB**. If you forget `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, every embedding is computed on color-channel-swapped pixels — the model still produces 768-dim vectors and the pipeline still runs, so there is no error. But the embedding distribution shifts (especially for skin tones, sky, vegetation, blood, vehicle paint), which silently degrades delta quality. The opposite mistake — converting twice — is equally invisible.

**Why it happens:**
- OpenCV is the only major Python imaging library that uses BGR by default. Developers coming from PIL/torchvision assume RGB.
- The bug is silent: no exception, no shape mismatch, no NaN. Reels still get produced. They just look slightly worse than they should.
- If the developer feeds frames into both `preprocess(Image.fromarray(...))` (which expects RGB) and a debug `cv2.imshow` (which expects BGR), one of the two will look wrong — and if they fix the one they can see (the display), they will break the one they cannot.

**How to avoid:**
- One-line guard: immediately after `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, assert nothing else swaps channels downstream. The conversion happens **exactly once**, in `extract.sample_frames()`, before the frames leave that function.
- Sanity test: pick a known-orange object frame from a sample video, embed it, then embed `frame[:, :, ::-1]`. Cosine similarity should be ~0.85–0.95 for natural images — if it's >0.99, channels were not swapped (you embedded the same array twice); if it's <0.5, your convert happened twice somewhere.
- Type discipline: `extract.sample_frames` should return `frames: list[np.ndarray]` documented as "RGB uint8 H×W×3". Anywhere that consumes `frames` should not call `cvtColor` again.

**Warning signs:**
- Reels look "fine but not great" with no obvious bug — especially blue-dominant outdoor scenes scoring lower than they should and red/orange interior scenes scoring higher (BGR amplifies the blue channel of CLIP's response).
- Per-module test §0.5 won't catch this — embeddings are still L2-normalized and shape `(N, 768)`. Add a channel-order sanity test as part of `extract.py` verification.

**Phase to address:** Phase 1 (`extract.py`) — verification step, before any deltas are computed.

---

### Pitfall 2: `CAP_PROP_POS_FRAMES` seek inaccuracy on VFR / long-GOP inputs

**What goes wrong:**
The spec's frame sampling code uses `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)`. On variable-frame-rate (VFR) MP4s and long-GOP H.264 inputs (which body cam footage often is), this seek lands on the **nearest preceding keyframe**, not the requested frame. Reported timestamp `frame_idx / video_fps` no longer matches the actual frame returned. Downstream, every clip timestamp is off by a fraction of a second to several seconds — and the error is non-uniform across the video, so you cannot just apply a constant offset.

**Why it happens:**
- `CAP_PROP_FPS` returns the *nominal* fps from the container header. For VFR video this is meaningless — frames are not equally spaced in time.
- OpenCV's FFmpeg backend implements `CAP_PROP_POS_FRAMES` by seeking to the byte offset of the nearest keyframe and then decoding forward, but the "decoding forward" part is inconsistent across OpenCV builds and FFmpeg versions.
- Body cam encoders frequently produce VFR output to handle dropped frames during high-motion or low-light, so this is the common case, not the edge case.

**How to avoid:**
- **Replace seek-by-frame with seek-by-time using `CAP_PROP_POS_MSEC`**, and read the **actual** timestamp back after each read:
  ```python
  cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
  ret, frame = cap.read()
  actual_ts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
  timestamps.append(actual_ts_sec)  # use the ACTUAL time, not the requested time
  ```
- Or, more robust: probe duration with `ffprobe -show_entries format=duration -of csv=p=0`, generate target timestamps, and use the actual returned timestamp from each `cap.get(CAP_PROP_POS_MSEC)`.
- Alternative if seeking proves unreliable: decode every frame with `cap.read()` in a loop and skip in software (decode count modulo `int(video_fps / 2)`). Slower, but fully accurate. Acceptable since this only runs once per video.

**Warning signs:**
- First-five-timestamps test in spec §0.5 passes (timestamps look like `[0.0, 0.5, 1.0, 1.5, 2.0]`) but spot-checking a clip in a video player shows the visual content is offset by 1–3 seconds from where the JSON claims.
- Number of returned frames is noticeably less than `expected = int(duration_sec * 2)` — e.g., requested 3,600 frames, got 3,210.
- Timestamps printed from `cap.get(CAP_PROP_POS_MSEC)` after each read drift from the requested grid by more than ~0.05s.

**Phase to address:** Phase 1 (`extract.py`) — must be solved before any embedding work, since wrong timestamps invalidate every downstream alignment.

---

### Pitfall 3: `CAP_PROP_FRAME_COUNT` lies, loop terminates early or runs past EOF

**What goes wrong:**
`int(cap.get(CAP_PROP_FRAME_COUNT))` is read from the container's metadata and is frequently **wrong** for body cam MP4s — sometimes off by hundreds of frames. The spec's loop `while frame_idx < total_frames: ... frame_idx += int(sample_interval)` either:
- terminates early (claimed count > actual count) — but the `if not ret: break` saves you, so this manifests only as fewer frames than expected, not a crash;
- or runs past EOF (claimed count > actual count, you believed `total_frames` for budget calculations) — `video_duration_sec = total_frames / video_fps` is wrong, the budget is wrong, and the JSON's `video_duration_sec` field disagrees with what `ffprobe` reports.

**Why it happens:**
- MP4 metadata frame counts depend on the muxer; if the file was truncated, repaired, or written by a body-cam-specific encoder, the count is unreliable.
- OpenCV does not validate the count against actual decodable frames at open time.

**How to avoid:**
- **Source of truth for duration is `ffprobe`, not OpenCV.** In `utils.get_video_duration()`:
  ```python
  out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration', '-of', 'csv=p=0', path],
                       capture_output=True, text=True, check=True)
  return float(out.stdout.strip())
  ```
- Drive the sampling loop with a `while True: ret, frame = cap.read(); if not ret: break` pattern (or `CAP_PROP_POS_MSEC` based, see Pitfall 2). Do not pre-compute the iteration count from `FRAME_COUNT`.
- Cross-check at end: `assert abs(timestamps[-1] - duration_from_ffprobe) < 1.0` — if the last sampled timestamp is more than a second short of the ffprobe duration, log a warning.

**Warning signs:**
- JSON's `video_duration_sec` differs from `ffprobe`-reported duration by more than 0.5s.
- `len(timestamps)` is very different from `int(duration_sec * 2)`.

**Phase to address:** Phase 1 (`extract.py`) — and `utils.py` for duration helper.

---

### Pitfall 4: `model.eval()` forgotten or undone after model load

**What goes wrong:**
open_clip's `create_model_and_transforms` returns a model in train mode by default in some versions. Without `model.eval()`, dropout and any train-mode batch-norm behavior remain active, so calling the same forward pass twice on the same input yields different embeddings. Cosine deltas will fluctuate frame-to-frame for static footage, MAD normalization will appear "noisy" with no real signal, and reproducibility across runs collapses.

**Why it happens:**
- The spec explicitly calls `model.eval()` (§2), but it is easy to drop the call when refactoring or when copy-pasting only the `create_model_and_transforms` line.
- Some torchvision wrappers re-enable train mode after `.to(device)` on certain torch versions — defensive code calls `.eval()` after `.to()`, not before.

**How to avoid:**
- Place `model.eval()` immediately before the inference loop, not just after model creation. Defensive, idempotent.
- Wrap inference in `with torch.inference_mode():` (preferred over `torch.no_grad()` — slightly faster and asserts no autograd tracking). If grads were silently being tracked, memory will balloon.
- Determinism test: embed the same frame twice in the same process, assert exact equality:
  ```python
  e1 = embed_one(frame); e2 = embed_one(frame)
  assert np.array_equal(e1, e2), "Model is non-deterministic — eval() not set or dropout active"
  ```

**Warning signs:**
- Running the pipeline twice on the same video produces different selected clips.
- Static-footage windows (e.g., a 30-second clip of an officer standing still) show non-trivial MAD-normalized scores instead of being floored at zero.
- Per-frame cosine deltas have a noisy floor of ~0.01–0.03 instead of ~0.001–0.005.

**Phase to address:** Phase 1 (`extract.py`) — embedding inference function.

---

### Pitfall 5: `preprocess` applied to wrong type (ndarray vs PIL vs tensor)

**What goes wrong:**
open_clip's `preprocess` returned by `create_model_and_transforms` is a `torchvision.transforms.Compose` that begins with `Resize` and `CenterCrop` — these expect a **PIL Image**, not a numpy array or torch tensor. Passing a numpy ndarray raises `TypeError: img should be PIL Image. Got <class 'numpy.ndarray'>`. Some developers "fix" this by wrapping the ndarray in `torch.from_numpy(...)` and reshaping to `(C, H, W)` — which silently bypasses `Resize` and `CenterCrop`, so you embed full-resolution non-square crops, producing garbage embeddings.

**Why it happens:**
- The error message after the wrong fix is not raised — `Compose` accepts whatever the first transform's output is and tries to pass it to the next. With manual tensorization, the resize is skipped and the model receives an arbitrary-shape tensor that it processes (since CLIP-ViT can handle different spatial sizes via positional embedding interpolation in some implementations) but produces meaningless output.

**How to avoid:**
- Standard pattern, exactly:
  ```python
  from PIL import Image
  pil = Image.fromarray(frame_rgb)  # frame_rgb is uint8 H×W×3 RGB
  tensor = preprocess(pil)           # → (3, 224, 224) float32 normalized
  batch = torch.stack([preprocess(Image.fromarray(f)) for f in frame_batch]).to(device)
  ```
- Shape assertion before model call:
  ```python
  assert batch.shape[1:] == (3, 224, 224), f"preprocess produced wrong shape: {batch.shape}"
  ```

**Warning signs:**
- L2-norm test passes (vectors are still normalized at the head).
- But embedding deltas for visibly different frames are tiny (<0.05) — the model is producing collapsed/degenerate embeddings.
- Or: a `TypeError` from `transforms.Resize` early in development. Fix by `Image.fromarray`, NOT by manual tensorization.

**Phase to address:** Phase 1 (`extract.py`) — embedding inference function, with shape assertion as part of §0.5 testing.

---

### Pitfall 6: L2-normalization assumed but not enforced

**What goes wrong:**
The spec §2 specifies "L2-normalized embedding vectors of dimension 768". `open_clip.create_model_and_transforms` returns a model whose `encode_image` does **not** normalize by default — it returns raw projection outputs. The spec's delta computation assumes `embeddings` are unit vectors so cosine similarity reduces to a dot product. If embeddings are not normalized, dot products produce values outside [-1, 1], `1 - dot_products` produces nonsensical deltas (some negative, some > 2), and `np.clip(dot_products, -1.0, 1.0)` masks the bug — clipping a value of 47 to 1 produces delta=0, so identical frames look identical and very different frames *also* look identical. The signal flatlines.

**Why it happens:**
- Many tutorials show `model.encode_image(x)` and call cosine similarity without normalization. Some open_clip examples normalize, others don't.
- The §0.5 test `np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)` catches this — but only if it is actually run, and only if the developer doesn't comment it out when "the assert is failing because of float precision" (it isn't, the embeddings genuinely aren't normalized).

**How to avoid:**
- Enforce explicitly inside the embedding function:
  ```python
  with torch.inference_mode():
      feats = model.encode_image(batch)
      feats = feats / feats.norm(dim=-1, keepdim=True)  # explicit L2 norm
  ```
- Keep the §0.5 assertion as a hard check, not a print:
  ```python
  norms = np.linalg.norm(embeddings, axis=1)
  assert np.allclose(norms, 1.0, atol=1e-5), f"Not normalized: norms in [{norms.min()}, {norms.max()}]"
  ```

**Warning signs:**
- §0.5 print step shows raw delta values either far above 0.5 (sometimes negative, sometimes >1) or suspiciously near zero everywhere — both indicate non-unit embeddings interacting with the clip.
- Plotted delta signal looks like white noise around a constant rather than a clear baseline with sparse spikes.

**Phase to address:** Phase 1 (`extract.py`) — embedding inference, with hard assertion in §0.5 verification.

---

### Pitfall 7: Batch vs single-frame inference produces different embeddings

**What goes wrong:**
For most CLIP implementations, `encode_image(batch_of_32)` and stacking 32 individual `encode_image(single_frame)` calls produce **bit-identical** outputs. But not always — if the pipeline uses any layer with batch-dependent behavior (some normalization variants, deterministic-mode-off matmul), or if mixed precision (`torch.float16` autocast) is used in batched mode but not in single-frame mode, embeddings will differ at the 1e-4 level. That seems harmless, but cosine deltas of order 1e-3 (static footage baseline) are sensitive to it. A debug single-frame inference vs production batched inference can disagree on which peaks are detected.

**Why it happens:**
- Mixed precision: `with torch.autocast(...)` for the batched path is a common optimization. Single-frame debug paths often don't bother. The two paths now use different precision.
- cuDNN benchmark mode (`torch.backends.cudnn.benchmark = True`) selects different algorithms for different batch sizes.

**How to avoid:**
- Single batched code path. Even debug "embed this one frame" goes through the same function with `frames=[frame]`.
- Avoid `torch.autocast` for the spec's stated batch size of 32 unless verified to produce identical outputs.
- Set `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` at module init for the prototype (we are not optimizing throughput).

**Warning signs:**
- Reproducibility test (run pipeline twice on the same video) produces different clip selections by ~1–2 clips.
- Selected peaks at different ranks across runs.

**Phase to address:** Phase 1 (`extract.py`) — single inference path, plus reproducibility config in `pipeline.py` initialization.

---

### Pitfall 8: Off-by-one in delta-to-timestamp alignment (verification protocol)

**What goes wrong:**
The spec §4 says "delta[i] corresponds to timestamps[i+1]" (the change *arriving at* frame i+1). After median filtering and MAD normalization, the resulting `scores` array still has length N-1. When `select_peaks(scores, timestamps, ...)` is called, naive code passes `timestamps` (length N) and indexes `timestamps[peak_idx]` — but `peak_idx` is an index into the length-(N-1) score array. If the score-to-timestamp map is wrong by one, every reported peak time is shifted by 0.5s. This is invisible unless you can A/B against ground truth — and the spec acknowledges there is no ground truth.

The spec calls out the pitfall but does not say *how to verify it*. Below is the verification.

**Why it happens:**
- The mapping `score_index i → timestamps[i+1]` is mentioned in prose, not enforced by data structure. Once a score array is in numpy form, its index is just `0..N-2` with no metadata about which frame it corresponds to.
- After PELT changepoints (which are also indices into the smoothed signal of length N-1), the same off-by-one applies.

**How to avoid:**
- **Single helper function for the conversion**, used everywhere:
  ```python
  def score_index_to_timestamp(score_idx: int, frame_timestamps: np.ndarray) -> float:
      """delta/score at index i ⟺ frame i+1's timestamp (the 'arriving at' convention)."""
      return float(frame_timestamps[score_idx + 1])
  ```
  Never compute `timestamps[peak_idx]` inline. Always go through this function.
- **Synthetic verification test** (this is the answer to "how do you actually verify it"):
  1. Construct a fake video by interleaving two visually distinct frames (e.g., two solid colors) with a transition at known frame index `K`. Sample at 2fps to get 20 frames where frames 0..K-1 are color A and frames K..19 are color B.
  2. Compute deltas → scores. Expect exactly one large delta at score-index `K-1` (the transition between frame K-1 and frame K).
  3. Map to timestamp via the helper: `expected_ts = (K) * 0.5` (since frame K is the first "arrived at" frame of the new color).
  4. Assert `abs(score_index_to_timestamp(peak_idx, timestamps) - expected_ts) < 0.01`.
- Visual sanity check on real video: take the highest-scoring clip, open in a video player, scrub to the reported `peak_timestamp_sec`. The frame at that timestamp should be visually different from the frame ~0.5s earlier. If the visual change is at `peak_timestamp_sec - 0.5`, you're off by one.

**Warning signs:**
- All clip start/end times are systematically 0.5s later than where the visible event happens. Pad sec=3.0 hides this somewhat, but the asymmetry is still detectable.
- JSON `peak_timestamp_sec` is exactly 0.5s past the visible event for every clip across every video.

**Phase to address:** Phase 2 (`signal_processing.py`) — alignment helper + synthetic test. The synthetic test is the *answer* to the spec's call-out.

---

### Pitfall 9: `medfilt` edge effects produce phantom peaks at signal boundaries

**What goes wrong:**
`scipy.signal.medfilt(x, kernel_size=5)` zero-pads the input at both ends. For a non-zero baseline signal, the padded zeros pull the first 2 and last 2 samples of the median-filtered output downward — and crucially, when followed by MAD normalization, those depressed values cause the *adjacent* samples to look like positive anomalies. The first ~3 seconds and last ~3 seconds of every video can produce phantom peaks unrelated to actual content.

**Why it happens:**
- `medfilt` does not have a `mode` parameter for boundary handling — it always zero-pads. (`scipy.ndimage.median_filter` does accept `mode='reflect'`, but the spec uses `scipy.signal.medfilt`.)
- The MAD-normalization rolling window at the boundary is asymmetric (the spec's loop uses `start = max(0, i - half)`), which compounds the artifact: at i=0, the window is half-width and biased toward the spurious low values from edge-padding.

**How to avoid:**
- Use `scipy.ndimage.median_filter(x, size=5, mode='nearest')` (or `mode='reflect'`) instead of `scipy.signal.medfilt`. Behaviorally equivalent in the interior, much better at boundaries.
- Or: pad the signal manually with edge values before `medfilt`:
  ```python
  padded = np.concatenate([np.full(2, x[0]), x, np.full(2, x[-1])])
  smoothed = medfilt(padded, kernel_size=5)[2:-2]
  ```
- Alternative: discard the first/last 3 samples of the score array (1.5s of edge) from peak detection. Cheap if the videos are >3 minutes.

**Warning signs:**
- A reel that has its first or last clip at the very beginning or very end of the video, especially when the video starts with a static intro card or ends with the camera being holstered.
- §0.5 plot of raw vs smoothed deltas shows a dip at the first/last 2 samples of the smoothed signal.

**Phase to address:** Phase 2 (`signal_processing.py`) — median filter step.

---

### Pitfall 10: MAD = 0 windows in static footage poison normalization

**What goes wrong:**
The spec's `mad_normalize` guards against `mad == 0` with `if mad > 1e-8`. But the more dangerous case is `mad` very small but nonzero (e.g., 1e-6) — the guard doesn't trigger, and `(signal[i] - local_median) / mad` produces astronomical scores (10⁵+). These get clipped to 10 by the post-normalization clip, but every score in the window pegs to the ceiling, so they tie and `find_peaks` arbitrarily orders them by index. A static window of an officer standing still followed by a tiny shake produces a "highlight" of the tiny shake at full ceiling score, beating genuine high-action clips elsewhere in the video.

**Why it happens:**
- `median_abs_deviation` of mostly-identical values is dominated by float precision noise, not zero. The `> 1e-8` guard is too lenient.
- The clip to 10.0 hides the bug — you don't see the `inf` or `1e6` value, you see 10.0, which looks reasonable.

**How to avoid:**
- Raise the floor: `if mad > 1e-3` (interpretable: "if local variability is below 1e-3 cosine-distance units, treat the window as static"). 1e-3 is well below typical body cam baseline noise (~0.01) but well above float precision artifacts.
- Or, gate by the local median magnitude: `if mad < 0.01 * max(local_median, 1e-3): scores[i] = 0.0`.
- Diagnostic logging: count the number of frames where the zero-MAD branch fires. If >5% of the video is in zero-MAD windows, the static-footage issue is dominating and the window length might be wrong.

**Warning signs:**
- A reel from a body cam that includes long parking-lot or office segments selects clips from inside those segments (where nothing visible happens) at high MAD scores.
- Histogram of final scores has a noticeable spike at exactly the clipping ceiling (10.0).
- §0.5 print-min-max-mean test shows max=10.0 but mean is also high (>1.0), suggesting many frames are pegged.

**Phase to address:** Phase 2 (`signal_processing.py`) — MAD normalization, with diagnostic count printed in §0.5 verification.

---

### Pitfall 11: ffmpeg `-ss` before `-i` vs after `-i` semantics

**What goes wrong:**
The spec §7 places `-ss` **before** `-i`. With `-c copy`, this is "input seek": ffmpeg seeks to the **nearest preceding keyframe** at or before `-ss`, then begins copying. If the nearest keyframe is, say, 2 seconds before the requested start, the resulting clip begins 2 seconds *earlier* than intended — but the clip's internal timestamp is rebased to 0, so when you later concatenate, the visible start is wrong but the clip's metadata duration matches `(end - start)` mostly. You get clips that show content from 2s before the event.

If you instead place `-ss` **after** `-i`, it becomes "output seek": ffmpeg decodes from the start, discards frames until `-ss`, and starts copying — but with `-c copy` this combination silently produces a clip whose first second is empty/black or whose duration is shorter than requested, depending on the encoder.

The two are not interchangeable, and the spec's choice is correct *for stream copy* — but the imprecision is real.

**Why it happens:**
- `-ss` semantics are one of the most-asked questions in ffmpeg's mailing list. The behavior changed between FFmpeg 2.x and 4.x; old Stack Overflow answers contradict current behavior.
- The spec acknowledges ~1s imprecision is "acceptable," but does not warn that the imprecision is **directional** (always earlier, never later) — so a clip with `start=142.5, end=158.5` may visually show 140.5–158.5 of the source.

**How to avoid:**
- Keep `-ss` before `-i` (correct for `-c copy`), but **expand the JSON `start_sec` to reflect what the keyframe-aligned reality will be**, not the requested value. Or, alternatively, accept the imprecision and document it (spec already does).
- For the *manifest* to be honest: the JSON should report the requested padded window (`start_sec = peak - padding`), and an additional optional field could record the actual keyframe-aligned start probed via `ffprobe -show_frames` — but spec §8 doesn't include it, so leave it.
- If precise cuts ever become needed: drop `-c copy`, use `-c:v libx264 -crf 18 -c:a copy`, and switch to `-ss` *after* `-i` for accurate seeking. Spec §7 footnote already mentions this fallback.
- Hard test: extract a clip with `start_sec=10.0`, then probe with `ffprobe -show_packets -select_streams v:0 -of json clip.mp4` and check the first packet's `pts_time`. If it's >1s before 10.0 on every clip, keyframes are sparse — reels will look "early."

**Warning signs:**
- Clips visibly start 1–3 seconds before the action (you see an officer walking, then arriving at a scene, then the action).
- `ffprobe` of an extracted clip reports a duration noticeably longer than `end_sec - start_sec`.

**Phase to address:** Phase 4 (`export.py`) — clip extraction.

---

### Pitfall 12: concat demuxer rejects clips with different timebases or codec parameters

**What goes wrong:**
The concat demuxer (`-f concat`) requires every input file to have **identical** codec, codec parameters (resolution, pixel format, profile, level, time base, frame rate). When the source body cam video is itself well-formed, all extracted `-c copy` clips will share parameters and concat works. But:
- If one clip is extracted from a portion of the video that crosses a parameter change (rare in body cams, but happens at file-join boundaries on devices that split recordings),
- Or if any clip becomes empty/zero-byte due to keyframe alignment failure (Pitfall 11),
- Or if `-c copy` produces a clip with a slightly different time base because ffmpeg rounded `-ss` differently across calls,

the concat demuxer fails with `Non-monotonous DTS in output stream` warnings, then either produces a broken final reel (frozen frames at clip boundaries, audio drift, A/V desync) or fails outright with `Could not write header for output file`.

**Why it happens:**
- The concat demuxer is designed for files produced by the same encode in the same session. ffmpeg's `-c copy` is mostly compliant but not strictly so.
- Body cam videos sometimes have parameter changes mid-file (variable bitrate, dynamic resolution).

**How to avoid:**
- **Validate clips before concat:**
  ```python
  for clip in clip_paths:
      out = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', clip],
                           capture_output=True, text=True)
      assert os.path.getsize(clip) > 0, f"Zero-byte clip: {clip}"
      # Optionally parse JSON and check codec/resolution/pix_fmt match across all clips
  ```
- **Fallback to concat-by-re-encode** when the demuxer fails:
  ```python
  # If concat demuxer raises, retry with concat filter:
  filter_input = ''.join(f'-i {p} ' for p in clips)
  filter_complex = ''.join(f'[{i}:v][{i}:a]' for i in range(len(clips))) + f'concat=n={len(clips)}:v=1:a=1[outv][outa]'
  # produces a re-encoded but reliable concat
  ```
- **Use the more permissive concat protocol** (`concat:file1|file2|...`) only as last resort — it's TS-format only.

**Warning signs:**
- ffmpeg stderr contains "Non-monotonous DTS in output stream" or "Could not find codec parameters."
- Final reel plays but has frozen frames or audio dropouts at clip boundaries.
- Final reel duration in `ffprobe` is shorter than the sum of input clip durations.

**Phase to address:** Phase 4 (`export.py`) — concat step, with pre-concat validation pass.

---

### Pitfall 13: concat manifest path quoting / `-safe 0` interaction

**What goes wrong:**
The spec writes the concat manifest with `f.write(f"file '{os.path.abspath(path)}'\n")`. Two failure modes:
1. If any clip path contains a single quote (rare for `clip_001.mp4` but possible for video names with apostrophes like `Officer_O'Brien.mp4`), the manifest line breaks and the demuxer reports `Unsafe file name`.
2. The `-safe 0` flag is required to allow absolute paths — without it, ffmpeg refuses absolute paths or paths with `/` for security reasons. Spec correctly includes `-safe 0`, but if a developer copy-pastes the concat command into a debug shell session and drops the flag, debugging gets confusing.

**Why it happens:**
- The concat demuxer's manifest format escapes single quotes by writing `'\''` (yes, that exact four-character sequence). Most code doesn't bother because most filenames don't contain quotes.
- `-safe 0` is on by default in some ffmpeg builds and not others.

**How to avoid:**
- Sanitize filenames: replace `'` with `_` in clip filenames generated for export. Body cam video names are derived from `Path(video).stem` which can contain anything.
- Or, more robustly, use the `concat` filter form (no manifest file):
  ```python
  inputs = sum([['-i', p] for p in clips], [])
  fc = ''.join(f'[{i}:v][{i}:a]' for i in range(len(clips))) + f'concat=n={len(clips)}:v=1:a=1[v][a]'
  cmd = ['ffmpeg', '-y', *inputs, '-filter_complex', fc, '-map', '[v]', '-map', '[a]', output]
  ```
  This re-encodes (slower) but eliminates manifest fragility. Use as fallback only.
- Never construct manifest paths with relative paths — always `os.path.abspath`.

**Warning signs:**
- ffmpeg stderr: `Unsafe file name` or `Impossible to open` on a path that exists.
- Manifest file has multiple `file '...'` lines that look syntactically fine but ffmpeg can't parse one of them.

**Phase to address:** Phase 4 (`export.py`) — sanitize clip paths, hardcode `-safe 0`.

---

### Pitfall 14: torch / numpy / scipy nondeterminism affects peak ordering at score ties

**What goes wrong:**
`scipy.signal.find_peaks(scores, distance=...)` uses a deterministic algorithm, but its behavior on ties (two adjacent local maxima with equal score) depends on iteration order and the sample values' bit-exact representation. If embeddings differ by 1e-7 across runs (due to nondeterministic GPU matmul, parallel reductions, or `torch.use_deterministic_algorithms(False)` defaults), score ties resolve differently, the set of peaks differs by 1–2 entries, and the greedy budget fill picks slightly different clips.

**Why it happens:**
- PyTorch matmul on GPU is non-deterministic by default for some kernels (cuBLAS/cuDNN).
- numpy operations involving thread-parallel BLAS (OpenBLAS, MKL) can produce bit-different sums depending on thread scheduling.
- The `--height` cutoff at `1.5` will admit/reject a peak that scores near `1.5000001` based on the run.

**How to avoid:**
- At pipeline entry:
  ```python
  import os
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic cuBLAS
  os.environ['OMP_NUM_THREADS'] = '1'                # deterministic numpy reductions
  import torch
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.manual_seed(0)
  np.random.seed(0)
  ```
- Tolerate the residual by widening the `--height` margin: peaks within 5% of the threshold are inherently fragile; document this rather than fight it.
- Reproducibility test as part of acceptance: run pipeline twice on the same video, assert byte-identical JSON output (or at minimum, identical clip start/end timestamps to 0.01s).

**Warning signs:**
- Reel JSON differs across runs: same number of clips, slightly different start_sec/end_sec values, sometimes one clip dropped.
- `mad_score` values in JSON have last-digit differences across runs.

**Phase to address:** Phase 5 (`pipeline.py`) — global determinism setup at entry, plus repro test at end.

---

### Pitfall 15: Lazy GPU init on first inference perturbs first batch's embeddings

**What goes wrong:**
On first GPU call, CUDA/Metal allocates context, compiles kernels (cuDNN benchmark), and warms caches. The first batch's inference can be subtly different from steady-state — usually identical numerically, but if `torch.backends.cudnn.benchmark = True` is on (default in some setups), the kernel chosen for the first batch may differ from later batches based on workload, producing 1e-6 numerical drift between batch 1 and batch 2. Across the same video, that's invisible. Across re-runs (where warm-up state differs), it compounds Pitfall 14.

**Why it happens:**
- CUDA lazy init is real and cannot be fully disabled without `CUDA_LAUNCH_BLOCKING=1`.
- cuDNN benchmark mode picks a different algorithm for the same shape based on prior memory allocation history.

**How to avoid:**
- Explicit warm-up before the real loop:
  ```python
  with torch.inference_mode():
      _ = model.encode_image(torch.zeros(1, 3, 224, 224, device=device))
  ```
- `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` (also helps Pitfall 14).
- For CPU-only (the spec's stated target), this pitfall is mostly absent — so the impact is limited to GPU-equipped reviewer machines.

**Warning signs:**
- First batch's mean embedding-norm differs from other batches' by >1e-6 (after the explicit normalize).
- First few seconds of any video have systematically different delta noise floor than the rest.

**Phase to address:** Phase 1 (`extract.py`) — embedding inference function.

---

### Pitfall 16: JSON float precision drift breaks reproducibility byte-comparison

**What goes wrong:**
`json.dump({"start_sec": 142.50000000000003, ...})` emits the full float repr by default. Across runs with any nondeterminism (Pitfall 14, 15), the trailing digits differ, so a `diff` between two runs' JSONs shows changes even when "the answer is the same." Reviewer running `python pipeline.py video_01.mp4` twice and seeing a non-empty diff loses confidence in the system.

**Why it happens:**
- `start_sec`, `mad_score`, `raw_cosine_delta` are all double-precision floats. Default `json.dump` emits `repr(x)` which can be 17 significant digits.
- Embedding-derived values (`raw_cosine_delta`, `mad_score`) are particularly sensitive.

**How to avoid:**
- Round on the way out:
  ```python
  def round_sec(x): return round(float(x), 3)   # millisecond precision
  def round_score(x): return round(float(x), 4) # 4 sig figs for scores
  ```
- Apply consistently in JSON construction. The schema will then be byte-stable across deterministic re-runs.
- Use `json.dump(..., sort_keys=True, indent=2)` for stable key ordering.

**Warning signs:**
- `diff json1 json2` shows changes in trailing digits despite identical clip selections.

**Phase to address:** Phase 5 (`pipeline.py`) — JSON emit step.

---

### Pitfall 17: `coincides_with_pelt_changepoint` field omitted vs null vs false

**What goes wrong:**
Spec §5 says "set `coincides_with_pelt_changepoint` to `null` when PELT is disabled." Three ways this gets wrong:
1. **Field omitted entirely** when `--pelt` is off (the developer skips writing it). Downstream consumers of the JSON crash on `clip['coincides_with_pelt_changepoint']` with KeyError.
2. **Set to `false`** (Python `False`) when PELT is off. Now the field is present but its value lies — `false` should mean "PELT was run, peak does not coincide," not "PELT was not run."
3. **Set to `null`** when PELT is on but no changepoints were detected (e.g., PELT raised, was caught silently). Same lie in the other direction.

**Why it happens:**
- Three states (disabled, ran-and-no, ran-and-yes) collapse easily into two during refactoring.
- Python's `None` serializes to JSON `null`; `False` serializes to `false`. Easy to swap.

**How to avoid:**
- Single rule, hardcoded once: `coincides = None if not use_pelt else (any(abs(idx - cp) <= 5 for cp in changepoints))`.
- JSON schema check at write time: assert every clip has the key, and its value is `None` iff `--pelt` was not passed:
  ```python
  for clip in clips_json:
      assert 'coincides_with_pelt_changepoint' in clip
      if not use_pelt:
          assert clip['coincides_with_pelt_changepoint'] is None
      else:
          assert clip['coincides_with_pelt_changepoint'] in (True, False)
  ```
- Unit-style sanity check: emit one JSON with `--pelt`, one without, on the same video. Diff them — should differ only in the `coincides_with_pelt_changepoint` field (None vs True/False) and possibly clip ordering due to score boost.

**Warning signs:**
- KeyError or TypeError when consuming the JSON downstream.
- `false` for every clip in a `--pelt`-off run (semantically wrong even if syntactically present).

**Phase to address:** Phase 5 (`pipeline.py`) — JSON schema construction, with pre-write validation.

---

### Pitfall 18: Greedy budget fill admits a tiny leftover clip via partial-clip logic

**What goes wrong:**
Spec §6 budget enforcement: if the next-by-score clip doesn't fit, allow a partial clip if `remaining >= 3.0`. Edge case: `remaining = 3.0` exactly, the partial clip is `(start, start + 3.0, score)` — a 3-second clip starting at the score's peak's padded start, which loses the actual peak content (the peak is centered at `start + padding`, and if padding=8 and remaining=3, you get the lead-in, not the peak). The reel ends with an uninformative clip showing approach to an event.

**Why it happens:**
- The partial-clip logic shortens from the *end*, not from around the peak. With adaptive padding of 8s on each side, the 3s remainder is cut from the front of the padding, including no actual peak content.

**How to avoid:**
- Center the partial clip on the peak instead:
  ```python
  if remaining >= 3.0:
      half_rem = remaining / 2
      partial_start = max(start, peak_time - half_rem)
      partial_end = min(end, peak_time + half_rem)
      selected.append((partial_start, partial_end, score))
  ```
  (Requires tracking `peak_time` alongside `(start, end, score)` through the merge step — note that merging discards the original peak. Alternative: store `peak_time` as a fourth tuple element through merging, propagating the higher-scoring peak's time.)
- Or: drop the partial-clip logic entirely if `remaining < padding * 2`. Cleaner.

**Warning signs:**
- The last clip in the reel (chronologically last *or* lowest-scoring depending on exact greedy order) shows a lead-in or aftermath rather than the moment the peak detected.

**Phase to address:** Phase 3 (`clip_selection.py`) — budget enforcement, with peak_time threaded through merging.

---

### Pitfall 19: `merge_clips` keeps highest score but loses peak alignment

**What goes wrong:**
Spec §6 merge step keeps `max(prev[2], score)` for the score of merged clips, but discards which peak that score came from. Downstream JSON `peak_timestamp_sec` field for a merged clip is ambiguous — which peak's timestamp do you report? If you report the start of the merged window, it's not a peak; if you report the highest-scoring peak's time, you need to track it through the merge.

**Why it happens:**
- The merge tuple is `(start, end, score)` — three values. The peak time is implicit (assumed to be the midpoint or the higher-scored member's center) but not stored.
- JSON schema requires `peak_timestamp_sec` per clip.

**How to avoid:**
- Carry peak time in the merge tuple from the start: `(start, end, score, peak_time)`.
- On merge, keep the peak_time of the higher-scoring source: `if score > prev[2]: prev[3] = peak_time`.
- Sanity check in JSON: assert `start_sec <= peak_timestamp_sec <= end_sec` for every clip.

**Warning signs:**
- JSON `peak_timestamp_sec` is outside the `[start_sec, end_sec]` range for some clips.
- Manual inspection of merged clips: the "peak" timestamp falls in the middle of the merged window, not at any visually-distinct moment.

**Phase to address:** Phase 3 (`clip_selection.py`) — merge step, with assert.

---

### Pitfall 20: `find_peaks` `distance` parameter measured in samples, not seconds

**What goes wrong:**
Spec §6 correctly converts: `min_gap_samples = int(min_gap_sec * fps)`. But `int(15.0 * 2.0) = 30`, while a developer passing `min_gap_sec=15` to a function whose argument-name memory has decayed might accidentally call `find_peaks(scores, distance=15)` — a 7.5-second min-gap instead of 15. Three times more peaks, all in the wrong density regime, budget enforcement compensates by selecting the 1–2 highest, reel looks "shorter than expected" with all clips clustered.

**Why it happens:**
- `find_peaks`'s `distance` parameter is in samples — undocumented in casual reading, easy to forget.
- The `--min-gap-sec` CLI flag is in seconds; the unit conversion happens once and then the result is passed to `find_peaks`. If anyone refactors this call without re-conversion, the bug is silent.

**How to avoid:**
- Wrap the conversion in a single function that takes seconds and an fps, returning samples. Type-hint with `Annotated[int, "samples"]` if the project standard supports it; otherwise comment.
- Defensive: `assert min_gap_samples == int(min_gap_sec * fps)` immediately before the `find_peaks` call.
- Sanity test: peaks returned should have `min(np.diff(peaks)) >= min_gap_samples`. Assert it after the call.

**Warning signs:**
- Number of peaks 2–3× expected based on spec defaults.
- Selected clips visually cluster around 1–2 events rather than spread across the video.

**Phase to address:** Phase 3 (`clip_selection.py`) — peak detection.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use `CAP_PROP_FRAME_COUNT` for duration | One-line, no ffprobe dependency | Wrong duration on VFR / repaired files; cascades into wrong budget and wrong JSON | Never — `ffprobe` is already a system dep via spec §1 |
| Skip `model.eval()` because "it's eval mode by default" | Saves one line | Nondeterministic embeddings, irreproducible reels | Never |
| Skip explicit L2 normalization because "open_clip says it's normalized" | Saves one line | Silent — clip-to-1 hides the bug; degraded delta quality | Never — assert + explicit norm is one line each |
| Use `scipy.signal.medfilt` (zero-pads) instead of `scipy.ndimage.median_filter` | Already in spec example code | Phantom peaks at video boundaries | Acceptable for prototype if first/last 1.5s are dropped from peak detection |
| Hard-code `mad > 1e-8` floor | Spec example | Static-footage windows poison normalization | Bump to `1e-3` — same one line |
| Skip determinism env vars | One less stanza in `pipeline.py` | JSONs differ across runs; reviewer loses trust | Never — five lines, copy-paste once |
| Round JSON floats to 3/4 decimals only at the end | Cleaner schema, byte-stable | Loses sub-millisecond precision (irrelevant) | Always — no real cost |
| Re-encode all clips with `libx264 -crf 18` | Eliminates Pitfalls 11, 12, 13 in one shot | 5–20× slower export; ~3 min/video on CPU | Only if `-c copy` produces a visibly broken reel on any sample |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenCV `VideoCapture` | Trust `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` for VFR video | Use `ffprobe` for duration; drive sampling loop until `read()` returns False |
| OpenCV → CLIP handoff | Pass numpy ndarray directly to `preprocess` | Wrap in `Image.fromarray(rgb_array)` first |
| open_clip `encode_image` | Assume output is L2-normalized | Explicit `feats / feats.norm(dim=-1, keepdim=True)` plus assertion |
| `scipy.signal.medfilt` boundary | Treat output as ready-to-use at edges | Drop edge samples or switch to `scipy.ndimage.median_filter(mode='nearest')` |
| `scipy.signal.find_peaks` `distance` | Pass seconds | Convert to samples: `int(sec * fps)` |
| ffmpeg `-ss` placement | Place after `-i` with `-c copy` | Place before `-i` for input seek (current spec is correct) |
| ffmpeg concat demuxer | Assume all `-c copy` outputs are concat-compatible | Pre-validate with `ffprobe`; fall back to concat filter on failure |
| ffmpeg manifest paths | Forget `-safe 0` or use relative paths | Always `os.path.abspath` and always `-safe 0`; sanitize quotes in stems |
| `subprocess.run` ffmpeg call | `capture_output=True` swallows errors silently | Always check `returncode`; log stderr on failure (spec uses `check=True` — keep it) |

## Performance Traps

This is a single-shot batch pipeline, so most "scale" concerns are absent. The relevant performance traps are:

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading frames into memory all at once | OOM on 30+ minute videos: 3,600 frames × 1080p × 3 bytes ≈ 22 GB | Stream frames batch-by-batch into the model; don't accumulate in `frames` list | >15 min video at full HD on 16 GB machine |
| Re-importing `ruptures` even when `--pelt` is off | ~2s import overhead per pipeline run | Import inside the `if use_pelt:` branch (spec already implies this) | Always — cheap to fix |
| Per-frame `cap.set` then `cap.read` instead of streaming | 5–10× slower on long videos due to keyframe seeking | For 2fps from 30fps source, decode-and-skip may actually be faster than seek-per-frame | All inputs over ~5 min |
| Rolling MAD computed in a Python `for` loop (spec example) | O(N × W) ≈ 3,600 × 180 = 650k median computations; ~30s per video | Use a sliding-median data structure or sklearn's `RollingMedian`; or precompute per-window medians via `scipy.ndimage.median_filter(size=180)` | All videos >10 min |
| Writing all clip files to disk before concat | 2× disk usage; also makes Pitfall 12 worse | Acceptable for prototype; the `output/clips/` directory is already in spec | Always acceptable for assignment scope |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces. Use this checklist at end of each phase.

- [ ] **Frame extraction:** Often missing — actual timestamps recorded from `CAP_PROP_POS_MSEC`, not requested. Verify by spot-checking 3 frames against video player scrubbing.
- [ ] **Embeddings:** Often missing — explicit L2 normalization (don't trust the model). Verify with `assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)`.
- [ ] **Embeddings:** Often missing — `model.eval()` after model creation. Verify by embedding the same frame twice; bytes must match exactly.
- [ ] **Delta computation:** Often missing — alignment helper used everywhere instead of inline `timestamps[idx+1]`. Verify with synthetic two-color-video test (Pitfall 8).
- [ ] **Median filter:** Often missing — boundary handling. Verify by plotting smoothed signal at first/last 5 samples and confirming no spurious dip.
- [ ] **MAD normalize:** Often missing — diagnostic count of zero-MAD-window samples. Verify by printing percentage; should be <5% on typical body cam footage.
- [ ] **Peak detection:** Often missing — `min(np.diff(peaks)) >= min_gap_samples` post-call assertion. Verify by running with default 15s gap and asserting.
- [ ] **Merge step:** Often missing — `peak_time` propagated through merge. Verify with `start_sec <= peak_timestamp_sec <= end_sec` assertion at JSON write time.
- [ ] **Budget enforcement:** Often missing — partial-clip logic centers on peak, not on the original padded start. Verify visually on a video where budget is barely exceeded.
- [ ] **Clip extraction (ffmpeg):** Often missing — `os.path.getsize(clip_path) > 0` check after each extract. Verify in `export.py` post-extract loop.
- [ ] **Concat (ffmpeg):** Often missing — fallback to concat-filter re-encode when concat-demuxer fails. Verify by intentionally breaking one clip's codec params and confirming pipeline recovers.
- [ ] **JSON output:** Often missing — `coincides_with_pelt_changepoint` is `None` (not `False`, not omitted) when `--pelt` is off. Verify by emitting JSON for the same video with and without `--pelt` and diffing.
- [ ] **JSON output:** Often missing — float rounding for byte-stable diffs. Verify by running pipeline twice and `diff`-ing the JSONs.
- [ ] **Reproducibility:** Often missing — determinism env vars (`OMP_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`). Verify with two-run-byte-identical-output test.
- [ ] **Reproducibility:** Often missing — explicit GPU warm-up batch before real inference. Verify by running on a GPU machine and confirming first-batch and Nth-batch embeddings of identical frames are bit-identical.
- [ ] **End-to-end:** Often missing — manual scrub through reel against original to confirm clip boundaries are off by less than ~2s (Pitfall 11). Spec §0.5 final step covers this; do it before declaring done.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| BGR/RGB swapped (Pitfall 1) | LOW | Re-run pipeline after fix; embeddings must be recomputed from scratch (cannot be patched) |
| Wrong timestamps from VFR seek (Pitfall 2, 3) | MEDIUM | Re-run with `CAP_PROP_POS_MSEC`-based sampling; downstream auto-corrects since timestamps drive everything |
| `model.eval()` missing (Pitfall 4) | LOW | Add the call, re-run. Embeddings change but pipeline structure unchanged |
| Not L2 normalized (Pitfall 6) | LOW | Add explicit normalize, re-run |
| Off-by-one alignment (Pitfall 8) | MEDIUM | Add helper, audit all callers; re-run JSON emit only (don't need to re-extract embeddings) |
| Edge-effect phantom peaks (Pitfall 9) | LOW | Switch to `ndimage.median_filter` or drop edge samples; re-run signal_processing onward |
| ffmpeg keyframe imprecision (Pitfall 11) | MEDIUM | Acceptable for prototype; otherwise fall back to libx264 re-encode |
| concat demuxer failure (Pitfall 12) | MEDIUM | Fall back to concat filter (re-encode); add the fallback as a try/except in `export.py` |
| Reproducibility differences across runs (Pitfall 14, 15) | LOW | Add determinism env vars at pipeline entry; re-run |
| JSON schema malformed (Pitfall 17, 19) | LOW | Patch JSON emit; do not need to re-run pipeline if intermediate state is preserved |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. BGR/RGB confusion | Phase 1 (`extract.py`) | Channel-swap A/B test on known-color frame |
| 2. `CAP_PROP_POS_FRAMES` seek inaccuracy | Phase 1 (`extract.py`) | Spot-check 3 timestamps against video player |
| 3. `CAP_PROP_FRAME_COUNT` lies | Phase 1 (`extract.py`, `utils.py`) | `ffprobe` duration matches OpenCV-computed within 0.5s |
| 4. `model.eval()` forgotten | Phase 1 (`extract.py`) | Embed same frame twice; bytes match |
| 5. `preprocess` wrong type | Phase 1 (`extract.py`) | `assert batch.shape[1:] == (3, 224, 224)` |
| 6. L2-normalization not enforced | Phase 1 (`extract.py`) | `np.allclose(norms, 1.0, atol=1e-5)` |
| 7. Batch vs single inference inconsistency | Phase 1 (`extract.py`) | Single code path; benchmark mode off |
| 8. Off-by-one timestamp alignment | Phase 2 (`signal_processing.py`) | Synthetic two-color-video test |
| 9. `medfilt` edge effects | Phase 2 (`signal_processing.py`) | Plot first/last 5 samples; no dip |
| 10. MAD = 0 windows | Phase 2 (`signal_processing.py`) | Count zero-MAD samples; should be <5% |
| 11. ffmpeg `-ss` semantics | Phase 4 (`export.py`) | `ffprobe` first packet pts of extracted clip |
| 12. Concat demuxer codec/timebase | Phase 4 (`export.py`) | Pre-concat `ffprobe` validation pass |
| 13. Manifest path quoting | Phase 4 (`export.py`) | Sanitize stems; absolute paths; `-safe 0` |
| 14. Torch/numpy nondeterminism | Phase 5 (`pipeline.py`) | Two-run-identical-JSON test |
| 15. Lazy GPU init | Phase 1 (`extract.py`) | Explicit warm-up batch |
| 16. JSON float precision drift | Phase 5 (`pipeline.py`) | `diff` two runs' JSONs |
| 17. `coincides_with_pelt_changepoint` schema | Phase 5 (`pipeline.py`) | Pre-write schema assertions |
| 18. Partial-clip not centered on peak | Phase 3 (`clip_selection.py`) | Visual inspection of last clip in reel |
| 19. Merge loses peak alignment | Phase 3 (`clip_selection.py`) | `start <= peak_ts <= end` assertion |
| 20. `find_peaks` distance unit confusion | Phase 3 (`clip_selection.py`) | Post-call `min(np.diff(peaks)) >= min_gap_samples` |

## Sources

- OpenCV documentation on `VideoCapture` properties — known limitations of `CAP_PROP_POS_FRAMES` and `CAP_PROP_FRAME_COUNT` on VFR/long-GOP MP4s (HIGH confidence; documented behavior)
- open_clip GitHub issues on inference reproducibility and `model.eval()` defaults (MEDIUM confidence; community-reported)
- scipy.signal.medfilt vs scipy.ndimage.median_filter boundary-handling docs (HIGH confidence; documented)
- ffmpeg-user mailing list and ffmpeg wiki on `-ss` placement semantics with `-c copy` vs `-c:v libx264` (HIGH confidence; documented and tested)
- ffmpeg concat demuxer documentation on identical codec parameter requirement (HIGH confidence; in official docs)
- PyTorch reproducibility documentation (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) (HIGH confidence; official docs)
- Spec §0.5 testing protocol and §11/§12 — this document complements them; pitfalls below are *not* covered there

---
*Pitfalls research for: visual-embedding-based body cam highlight extraction pipeline*
*Researched: 2026-05-06*
