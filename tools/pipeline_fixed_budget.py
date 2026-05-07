#!/usr/bin/env python3
"""Phase 6 SUPPLEMENTAL: produce 1-minute highlight reels per video,
regardless of source duration. Spec §6's default budget is
(duration/1800)*60 = 1 min per 30 min, which makes the test videos
(~3 min sources → ~6 s reels) too short to be useful for review.

This script is a sibling to tools/pipeline_from_cache.py — same logic,
same cache-bypass, but with two overrides:
  1. budget_sec is fixed at TARGET_BUDGET_SEC (default 60.0) regardless
     of source duration. compute_budget_seconds() is bypassed.
  2. Output goes to output/reels_60s/{video}_highlight_60s.mp4 and
     output/timestamps_60s/{video}.json so it doesn't overwrite the
     spec-compliant default-budget reels.

Frozen tuning parameters (Phase 5): --height 1.5 --min-gap-sec 15.0
                                    --merge-gap-sec 3.0
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/Users/aryanahuja/abel_police_interview")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np

from pipeline import _build_manifest
from signal_processing import (
    compute_deltas,
    smooth_deltas,
    mad_normalize,
    detect_changepoints,
    score_index_to_timestamp,
)
from clip_selection import (
    select_peaks,
    apply_pelt_boost,
    compute_padding,
    build_clips,
    merge_clips,
    enforce_budget,
)
from export import extract_clip, concat_clips
from utils import probe_video_metadata


def run_fixed_budget(
    video_path: str,
    *,
    target_budget_sec: float,
    use_pelt: bool,
    height: float,
    min_gap_sec: float,
    merge_gap_sec: float,
) -> None:
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem

    # Custom output dirs — sibling to spec-default output/.
    reels_60_dir = REPO / "output" / "reels_60s"
    timestamps_60_dir = REPO / "output" / "timestamps_60s"
    clips_60_dir = REPO / "output" / "clips_60s" / video_name
    cache_dir = REPO / "output" / "cache"
    for d in (reels_60_dir, timestamps_60_dir, clips_60_dir):
        d.mkdir(parents=True, exist_ok=True)

    emb_path = cache_dir / f"{video_name}_embeddings.npy"
    ts_path = cache_dir / f"{video_name}_timestamps.npy"
    if not emb_path.exists() or not ts_path.exists():
        print(f"[FATAL] missing fixture for {video_name}", flush=True)
        sys.exit(2)

    meta = probe_video_metadata(video_path_obj)
    video_duration_sec = float(meta["duration_sec"])

    # If the override budget exceeds the source duration, clamp.
    budget_sec = min(target_budget_sec, video_duration_sec)
    if budget_sec < target_budget_sec:
        print(f"[note] target {target_budget_sec}s exceeds source duration "
              f"{video_duration_sec:.1f}s; clamping to {budget_sec:.1f}s")

    print(f"[1/6] [CACHED] loading embeddings + timestamps...")
    embeddings = np.load(emb_path)
    timestamps = np.load(ts_path)
    print(f"[1/6] embeddings shape={embeddings.shape}  duration={video_duration_sec:.1f}s")
    print(f"[2/6] [CACHED] CLIP step skipped")
    print(f"[3/6] Computing embedding deltas...")
    raw_deltas = compute_deltas(embeddings)
    print(f"[4/6] Filtering and normalizing signal...")
    smoothed = smooth_deltas(raw_deltas)
    scores = mad_normalize(smoothed, window_samples=180)
    changepoints = detect_changepoints(smoothed, penalty=3.0) if use_pelt else None

    print(f"[5/6] Selecting clips with FIXED BUDGET = {budget_sec}s...")
    peak_indices, peak_scores = select_peaks(
        scores, timestamps, height=height, min_gap_sec=min_gap_sec, fps=2.0,
    )
    if use_pelt and changepoints is not None:
        peak_indices, peak_scores = apply_pelt_boost(
            peak_indices, peak_scores, changepoints, window=5, boost=1.2,
        )
    peak_time_to_idx = {
        score_index_to_timestamp(int(idx), timestamps): int(idx)
        for idx in peak_indices
    }

    padding_sec = compute_padding(budget_sec)
    clips = build_clips(peak_indices, peak_scores, timestamps,
                        video_duration_sec, padding_sec)
    merged = merge_clips(clips, gap_threshold_sec=merge_gap_sec)
    final_clips = enforce_budget(merged, budget_sec)
    total = sum(e - s for s, e, _, _ in final_clips)
    print(f"[5/6] {len(peak_indices)} peaks → {len(final_clips)} final clips; "
          f"total={total:.1f}s ({100*total/budget_sec:.1f}% of budget)")

    print(f"[6/6] Exporting highlight reel...")
    clip_paths: list[Path] = []
    for i, (start, end, _, _) in enumerate(final_clips):
        out_clip = clips_60_dir / f"{i:03d}.mp4"
        extract_clip(video_path_obj, out_clip, start, end)
        clip_paths.append(out_clip)
        print(f"[6/6] clip {i+1}/{len(final_clips)}  {start:.1f}..{end:.1f}s "
              f"-> {out_clip.name}")

    reel_path = reels_60_dir / f"{video_name}_highlight_60s.mp4"
    concat_clips([str(p) for p in clip_paths], str(reel_path), str(clips_60_dir))

    manifest = _build_manifest(
        video_path=video_path_obj,
        video_duration_sec=video_duration_sec,
        budget_sec=budget_sec,
        final_clips=final_clips,
        peak_time_to_idx=peak_time_to_idx,
        scores=scores,
        raw_deltas=raw_deltas,
        changepoints=changepoints,
    )
    json_path = timestamps_60_dir / f"{video_name}.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Done. Reel: {reel_path}")
    print(f"      JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 fixed-budget supplemental")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--target-budget-sec", type=float, default=60.0,
                        help="Fixed reel duration in seconds (default 60)")
    parser.add_argument("--pelt", action="store_true", default=False)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--min-gap-sec", type=float, default=15.0)
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    args = parser.parse_args()
    t0 = time.monotonic()
    run_fixed_budget(
        args.video,
        target_budget_sec=args.target_budget_sec,
        use_pelt=args.pelt,
        height=args.height,
        min_gap_sec=args.min_gap_sec,
        merge_gap_sec=args.merge_gap_sec,
    )
    print(f"Total wall-clock: {time.monotonic() - t0:.2f}s")
