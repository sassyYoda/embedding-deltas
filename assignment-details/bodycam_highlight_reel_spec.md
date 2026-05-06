# Body Cam Highlight Reel — MVP Specification
**Assignment:** AbelPolice Take-Home — Visual Embedding Highlight Reels  
**Author:** Aryan Ahuja  
**Framework:** Claude Code / Get Shit Done

---

## 0. Orientation

This document is the single source of truth for building the highlight reel pipeline. Every design decision has been made and justified. Claude Code should implement exactly what is described here — do not substitute alternatives unless a hard technical blocker is encountered, in which case flag it explicitly before proceeding.

The goal is a working end-to-end pipeline that:
1. Takes a raw body cam video as input
2. Produces a condensed highlight reel video as output
3. Emits a JSON file of timestamps and scores for each selected clip
4. Does all clip selection using only visual embeddings and frame timestamps — no transcript, no audio, no OCR, no LLMs

---

## 1. Environment & Dependencies

### Python version
Python 3.11+

### Required packages
```
torch
torchvision
open_clip_torch
opencv-python
numpy
scipy
ruptures
ffmpeg-python
tqdm
```

### ffmpeg
Must be installed at the system level (not just Python bindings). Verify with `ffmpeg -version`.

### Hardware assumption
CPU-only inference is acceptable for this prototype. If a GPU is available, open_clip will use it automatically. Do not add GPU-specific code paths — let open_clip handle device selection.

### Project structure
```
highlight_reel/
├── spec.md                  # this file
├── pipeline.py              # main pipeline — single entry point
├── extract.py               # frame sampling + embedding extraction
├── signal_processing.py     # delta computation, filtering, normalization
├── clip_selection.py        # peak detection, budget enforcement, merging
├── export.py                # ffmpeg clip cutting and concatenation
├── utils.py                 # shared helpers (video metadata, logging)
├── videos/                  # input videos (downloaded from Drive)
├── output/
│   ├── reels/               # final highlight reel .mp4 files
│   ├── clips/               # intermediate clip segments (can be deleted after concat)
│   └── timestamps/          # per-video JSON files with timestamps and scores
└── requirements.txt
```

---

## 2. Embedding Model

### Model
**CLIP ViT-L/14, OpenAI pretrained weights**

This is non-negotiable. Use open_clip with the OpenAI pretrained checkpoint specifically — this is the QuickGELU variant identified as most robust to domain-shift corruptions in Koddenbrock et al. (2025), outperforming SigLIP and ALIGN under MotionBlur and PerspectiveTransform in the Handheld domain — the closest analog to body cam footage in that benchmark.

### Loading
```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='openai'
)
model.eval()
```

### Inference
- Run in `torch.no_grad()` context
- Process frames in batches of 32 for memory efficiency
- Output: L2-normalized embedding vectors of dimension 768
- Do NOT fine-tune or modify the model in any way

### Caveat to note in writeup
The robustness paper evaluates classification stability under corruption, not embedding delta stability specifically. The assumption that classification robustness transfers to delta stability is plausible but unproven — flag this as a limitation.

---

## 3. Frame Sampling

### Rate
**2 fps (one frame every 0.5 seconds)**

### Justification
- At 30fps, consecutive frames are near-identical; embedding delta is dominated by sub-perceptual variation rather than meaningful scene change
- At <1fps, fast physical events (weapon draw, physical altercation) can unfold and resolve entirely between sampled frames, making them undetectable
- 2fps captures events resolving in under a second while keeping embedding extraction computationally feasible
- For a 30-minute video: 3,600 frames — manageable for local CLIP ViT-L/14 inference

### Implementation
Use OpenCV VideoCapture. Do not decode every frame — seek directly to sample timestamps.

```python
import cv2

def sample_frames(video_path: str, fps: float = 2.0):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    
    sample_interval = video_fps / fps  # frames between samples
    timestamps = []
    frames = []
    
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamps.append(frame_idx / video_fps)
        frame_idx += int(sample_interval)
    
    cap.release()
    return frames, timestamps
```

### Output
- `frames`: list of H×W×3 uint8 numpy arrays in RGB
- `timestamps`: list of float seconds corresponding to each frame

---

## 4. Delta Computation

### Method
Cosine distance between consecutive sampled frame embeddings.

```
delta(t) = 1 - cosine_similarity(embedding(t), embedding(t-1))
```

Cosine distance ranges from 0 (identical embeddings) to 2 (opposite embeddings). In practice for consecutive body cam frames expect values in the range 0.0 to ~0.5.

### Implementation
```python
import numpy as np

def compute_deltas(embeddings: np.ndarray) -> np.ndarray:
    """
    embeddings: (N, D) array of L2-normalized embedding vectors
    returns: (N-1,) array of cosine distances
    """
    # Since embeddings are L2-normalized, cosine similarity = dot product
    dot_products = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    # Clip to [-1, 1] for numerical safety before computing distance
    dot_products = np.clip(dot_products, -1.0, 1.0)
    return 1.0 - dot_products
```

### Note on timestamps alignment
After computing deltas there are N-1 values for N frames. Align delta(t) to the timestamp of the *later* frame in each pair — i.e., delta[i] corresponds to timestamps[i+1]. This is the natural choice: the delta represents the change *arriving at* frame i+1.

---

## 5. Signal Processing

This is the core of the approach. Apply in this exact order: **median filter → MAD normalization**. Do not swap the order.

### Why this order matters
If you normalize before filtering, isolated artifact spikes inflate the local MAD estimate and distort the normalization of surrounding frames. Filter first to get a clean signal, then normalize against the clean baseline.

### Step 1: Median Filter

**Kernel size: 5** (corresponding to a 2.5-second neighborhood at 2fps)

The median filter is used instead of Gaussian smoothing because:
- Body cam artifact spikes (compression glitches, momentary occlusion, lens flare) are impulse-type noise — isolated values that differ sharply from neighbors
- For impulse noise, the median filter is the principled choice: it replaces outlier values with the neighborhood median without propagating spike energy into neighboring samples
- Gaussian smoothing performs weighted averaging which blurs genuine event spikes alongside artifacts — counterproductive when sharp transitions are the detection target
- Citation: established signal processing literature (Huang et al., 1979; Tukey, 1977); the edge-preserving property of median filters for impulse noise is textbook knowledge

```python
from scipy.signal import medfilt

smoothed = medfilt(raw_deltas, kernel_size=5)
```

### Step 2: Rolling MAD Normalization

**Window: 90 seconds = 180 samples at 2fps**

Convert the smoothed delta into a locally-referenced anomaly score — number of median absolute deviations above the local median at each frame.

```python
from scipy.stats import median_abs_deviation

def mad_normalize(signal: np.ndarray, window_samples: int = 180) -> np.ndarray:
    """
    Rolling MAD normalization over a centered window.
    Returns score: how many MADs above local median each sample is.
    """
    scores = np.zeros(len(signal))
    half = window_samples // 2
    
    for i in range(len(signal)):
        start = max(0, i - half)
        end = min(len(signal), i + half)
        local = signal[start:end]
        
        local_median = np.median(local)
        mad = median_abs_deviation(local)
        
        if mad > 1e-8:  # guard against zero-MAD windows (static footage)
            scores[i] = (signal[i] - local_median) / mad
        else:
            scores[i] = 0.0
    
    return scores
```

**Why MAD over standard deviation:**
MAD is robust to the outliers it is designed to detect. Standard deviation gets inflated by large spikes, which suppresses their normalized score — the exact opposite of what you want. MAD uses median, which is resistant to outliers.

**Why 90-second window:**
- Too short (e.g., 10 seconds): a genuine high-action sequence elevates its own baseline, suppressing detection of events within it
- Too long (e.g., 10 minutes): slow global drifts (indoor/outdoor transition, lighting change) contaminate the local baseline
- 90 seconds captures local context without being dominated by global trends
- Directly motivated by Antonio's suggestion in email exchange

**Post-normalization clipping:**
```python
scores = np.clip(scores, 0.0, 10.0)
```
- Zero floor: suppress unusually calm moments, irrelevant to highlight detection
- Ceiling of 10: prevent single extreme transitions (camera off/on) from dominating clip ranking

### PELT changepoint detection (optional supplementary signal)

PELT is **opt-in only**. The pipeline must accept a `--pelt` boolean flag. When not passed, PELT is skipped entirely and the pipeline runs on MAD peaks alone. When passed, PELT runs as a cross-reference and boosts confidence scores for MAD peaks that coincide with changepoints.

**CLI flag:**
```bash
# Without PELT (default)
python pipeline.py videos/video_01.mp4

# With PELT
python pipeline.py videos/video_01.mp4 --pelt
```

**Argparse setup in pipeline.py:**
```python
parser.add_argument('--pelt', action='store_true', default=False,
                    help='Enable PELT changepoint detection as supplementary signal')
```

**Implementation when --pelt is passed:**
```python
import ruptures as rpt

def detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0):
    """
    Returns list of changepoint indices in the signal.
    penalty controls sensitivity — higher = fewer changepoints.
    """
    model = rpt.Pelt(model="rbf").fit(smoothed.reshape(-1, 1))
    # Last element is always len(signal) — exclude it
    changepoints = model.predict(pen=penalty)[:-1]
    return changepoints
```

When `--pelt` is active, boost the MAD score of any peak that falls within ±5 samples of a PELT changepoint by a fixed multiplier (1.2). This affects clip ranking but not the selection mechanism itself.

When `--pelt` is not active, `detect_changepoints()` is never called and `ruptures` does not need to be imported. The `coincides_with_pelt_changepoint` field in the JSON output should be set to `null` when PELT is disabled.

---

## 6. Clip Selection

### Duration budget
The assignment specifies: **max 1 minute of highlight reel per 30 minutes of source footage.**

```python
def compute_budget_seconds(video_duration_sec: float) -> float:
    return (video_duration_sec / 1800.0) * 60.0  # 1 min per 30 min
```

### Peak detection

Use scipy find_peaks on the MAD-normalized score signal.

```python
from scipy.signal import find_peaks

def select_peaks(scores: np.ndarray, timestamps: np.ndarray,
                 min_gap_sec: float = 15.0, fps: float = 2.0):
    """
    Find peaks in normalized score signal.
    min_gap_sec: minimum seconds between selected peaks.
    """
    min_gap_samples = int(min_gap_sec * fps)
    
    peaks, properties = find_peaks(
        scores,
        height=1.5,           # minimum MAD score to qualify
        distance=min_gap_samples
    )
    
    # Sort by score descending
    peak_scores = scores[peaks]
    sorted_idx = np.argsort(peak_scores)[::-1]
    sorted_peaks = peaks[sorted_idx]
    sorted_scores = peak_scores[sorted_idx]
    
    return sorted_peaks, sorted_scores
```

**height=1.5:** only flag moments at least 1.5 MADs above the local median. Conservative enough to suppress noise, permissive enough to catch moderate events.

**min_gap_sec=15.0:** prevents selecting 10 clips from one 30-second altercation. Each selected clip represents a distinct event.

### Clip construction with padding

Each selected peak becomes the center of a clip with padding on each side.

```python
def build_clips(peak_indices: np.ndarray, peak_scores: np.ndarray,
                timestamps: np.ndarray, video_duration_sec: float,
                budget_sec: float, padding_sec: float = 8.0):
    """
    Greedily add clips in score order until budget is exhausted.
    Returns list of (start_sec, end_sec, score) tuples.
    """
    clips = []
    total_duration = 0.0
    
    for idx, score in zip(peak_indices, peak_scores):
        peak_time = timestamps[idx]
        start = max(0.0, peak_time - padding_sec)
        end = min(video_duration_sec, peak_time + padding_sec)
        clip_duration = end - start
        
        if total_duration + clip_duration > budget_sec:
            # Try with reduced padding if we are close to budget
            remaining = budget_sec - total_duration
            if remaining >= 4.0:  # minimum viable clip
                end = min(video_duration_sec, start + remaining)
                clip_duration = end - start
            else:
                break
        
        clips.append((start, end, float(score)))
        total_duration += clip_duration
    
    # Sort clips chronologically for the final reel
    clips.sort(key=lambda x: x[0])
    return clips
```

**padding_sec=8.0:** 8 seconds before and after the peak. Gives enough context for a reviewer to understand what happened without inflating clip duration. Tune down if budget is tight.

### Merging overlapping clips

```python
def merge_clips(clips: list, gap_threshold_sec: float = 3.0):
    """
    Merge clips that overlap or are within gap_threshold_sec of each other.
    clips: sorted list of (start, end, score) tuples
    """
    if not clips:
        return []
    
    merged = [list(clips[0])]
    
    for start, end, score in clips[1:]:
        prev = merged[-1]
        if start <= prev[1] + gap_threshold_sec:
            # Extend previous clip
            prev[1] = max(prev[1], end)
            prev[2] = max(prev[2], score)  # keep highest score
        else:
            merged.append([start, end, score])
    
    return [tuple(c) for c in merged]
```

---

## 7. Export

### Per-clip extraction using ffmpeg

```python
import subprocess
import os

def extract_clip(input_path: str, output_path: str,
                 start_sec: float, end_sec: float):
    """
    Extract a clip from input video using stream copy (fast, no re-encode).
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-to', str(end_sec),
        '-i', input_path,
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

### Concatenation into final reel

```python
def concat_clips(clip_paths: list, output_path: str, temp_dir: str):
    """
    Concatenate clip files into a single highlight reel.
    Uses ffmpeg concat demuxer.
    """
    # Write concat manifest
    manifest_path = os.path.join(temp_dir, 'concat_manifest.txt')
    with open(manifest_path, 'w') as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', manifest_path,
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

**Note on `-c copy`:** stream copy avoids re-encoding, which is fast and lossless. The tradeoff is that cut points must align to keyframes — in practice this means clip boundaries may be slightly imprecise (within ~1 second). For this prototype that is acceptable. If precise cuts are required, replace `-c copy` with `-c:v libx264 -crf 18`.

---

## 8. JSON Timestamp Output

Each video must produce a JSON file at `output/timestamps/{video_name}.json` with this exact schema:

```json
{
  "video": "video_01.mp4",
  "video_duration_sec": 1842.3,
  "budget_sec": 61.4,
  "total_reel_duration_sec": 58.2,
  "embedding_model": "ViT-L-14 (OpenAI / QuickGELU)",
  "sampling_fps": 2.0,
  "clips": [
    {
      "clip_index": 0,
      "start_sec": 142.5,
      "end_sec": 158.5,
      "duration_sec": 16.0,
      "peak_timestamp_sec": 150.5,
      "mad_score": 4.23,
      "raw_cosine_delta": 0.187,
      "coincides_with_pelt_changepoint": true
    }
  ]
}
```

`coincides_with_pelt_changepoint` is true if the peak falls within ±5 samples of a PELT changepoint.

---

## 9. Main Pipeline Entry Point

`pipeline.py` should accept a video path as a command-line argument and run the full pipeline end to end.

```python
# pipeline.py
import argparse
import json
import os
from pathlib import Path

def run(video_path: str, use_pelt: bool = False):
    video_name = Path(video_path).stem
    os.makedirs('output/reels', exist_ok=True)
    os.makedirs('output/clips', exist_ok=True)
    os.makedirs(f'output/clips/{video_name}', exist_ok=True)
    os.makedirs('output/timestamps', exist_ok=True)

    print(f"[1/6] Sampling frames at 2fps...")
    # extract.py: sample_frames() → frames, timestamps

    print(f"[2/6] Extracting CLIP embeddings...")
    # extract.py: embed_frames() → embeddings

    print(f"[3/6] Computing embedding deltas...")
    # signal_processing.py: compute_deltas() → raw_deltas

    print(f"[4/6] Filtering and normalizing signal...")
    # signal_processing.py: medfilt() → smoothed
    # signal_processing.py: mad_normalize() → scores
    # if use_pelt: signal_processing.py: detect_changepoints() → changepoints
    #              boost scores at changepoint-coincident peaks by 1.2x
    # else: changepoints = None

    print(f"[5/6] Selecting clips...")
    # clip_selection.py: select_peaks() → peaks
    # clip_selection.py: build_clips() → clips
    # clip_selection.py: merge_clips() → merged_clips

    print(f"[6/6] Exporting highlight reel...")
    # export.py: extract_clip() for each clip
    # export.py: concat_clips() → final reel
    # Write JSON timestamps (coincides_with_pelt_changepoint = null if use_pelt is False)

    print(f"Done. Reel: output/reels/{video_name}_highlight.mp4")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--pelt', action='store_true', default=False,
                        help='Enable PELT changepoint detection as supplementary signal')
    args = parser.parse_args()
    run(args.video, use_pelt=args.pelt)
```

Run all 5 videos with:
```bash
for f in videos/*.mp4; do python pipeline.py "$f"; done
```

---


## 11. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | CLIP ViT-L/14 OpenAI | Most robust to MotionBlur/PerspectiveTransform in Handheld domain (Koddenbrock 2025) |
| Sampling rate | 2fps | Fast event capture vs. redundancy tradeoff |
| Delta metric | Cosine distance | Semantic similarity, robust to pixel-level noise |
| Smoothing filter | Median (kernel=5) | Edge-preserving for impulse noise; Gaussian would blur genuine spikes |
| Normalization | Rolling MAD (90s window) | Robust to outliers; contextualizes delta relative to local baseline |
| Peak threshold | 1.5 MAD | Conservative enough to suppress noise, permissive enough for moderate events |
| Min peak gap | 15 seconds | Prevents over-sampling a single event |
| Clip padding | 8 seconds either side | Sufficient context without inflating budget |
| Budget enforcement | Greedy by score rank | Prioritizes highest-confidence moments within duration constraint |
| Supplementary signal | PELT changepoints (opt-in via --pelt flag) | Cross-reference for increased confidence; does not drive selection; disabled by default |
| Cut method | ffmpeg stream copy | Fast, lossless, ~1s boundary imprecision acceptable for prototype |

---

## 12. Known Limitations to Acknowledge

List these explicitly in both the writeup and any verbal discussion with Antonio:

1. **Visually distinct ≠ important** — this is the fundamental limitation of the entire approach, not just an implementation issue
2. **No audio or transcript** — by design per assignment constraints, but these are the signals that would most improve performance
3. **Camera shake false positives** — MAD normalization mitigates but does not eliminate; residual shake during sustained movement may still trigger
4. **Lighting transitions** — walking indoors/outdoors produces high semantic delta unrelated to incident relevance
5. **Fixed MAD window** — adaptive window sizing would better handle footage with highly variable activity levels
6. **Classification→delta robustness gap** — the Koddenbrock paper validates classification stability, not delta signal stability specifically
7. **No labeled ground truth** — evaluation is qualitative (watch the reels); no quantitative precision/recall is possible without annotated body cam data

---

## 13. What Not To Do

- Do **not** use transcript, ASR, audio levels, OCR, or LLMs for clip selection — this violates the assignment constraints
- Do **not** manually select timestamps even to verify the pipeline — clip selection must be fully algorithmic
- Do **not** use a fixed global threshold instead of rolling MAD — this was the weakness of Antonio's original k-most-distinct approach
- Do **not** apply the median filter to the raw video frames — filter the 1D delta signal only, after cosine similarity is computed
- Do **not** re-encode clips unless stream copy produces visibly broken output — re-encoding is slow and unnecessary
- Do **not** over-polish — a working prototype with honest analysis beats a polished system with optimistic framing

---

*Build in this order: extract.py → signal_processing.py → clip_selection.py → export.py → pipeline.py. Get one video working end to end before touching the others.*
