#!/usr/bin/env python3
"""Cache-bypass Phase 5 harness — runs everything in pipeline.run() EXCEPT
the CLIP extraction (stages 1-2). Loads pre-computed embeddings + timestamps
from output/cache/{video}_{embeddings,timestamps}.npy and runs stages 3-6.

This is a TEMPORARY DEV-ONLY workaround for the MPS deadlock that hits
pipeline.py during stage 2 on tiger_woods (and would on marcus_jordan).
Validates that pipeline.py's signal/clip/export/JSON logic (stages 3-6 +
the JSON assembly with peak_idx recovery + three-state PELT field +
3dp/4dp rounding) all work end-to-end. The CLIP step itself was already
validated in Phase 1's standalone harness (extract.py).

NOT in the project layout. NOT a user-facing entry. Spec §9 contract is
unchanged — pipeline.py stays as the canonical CLI.

Usage: python pipeline_from_cache.py videos/tiger_woods.mp4 [--pelt]
                                     [--height H] [--min-gap-sec G]
                                     [--merge-gap-sec M]
"""
from __future__ import annotations

# Match pipeline.py's env-var stanza (D-68) so determinism applies.
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

# Reuse pipeline.py's helpers + Phase 1-4 module functions.
from pipeline import (
    _build_manifest,
    _round_seconds,
    _round_score,
    _compute_pelt_coincidence,
)
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
    compute_budget_seconds,
    compute_padding,
    build_clips,
    merge_clips,
    enforce_budget,
)
from export import extract_clip, concat_clips
from utils import probe_video_metadata, ensure_output_dirs


def run_from_cache(
    video_path: str,
    *,
    use_pelt: bool,
    height: float,
    min_gap_sec: float,
    merge_gap_sec: float,
) -> None:
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem

    paths = ensure_output_dirs(video_name)
    clips_dir = paths["clips"]
    reels_dir = paths["reels"]
    timestamps_dir = paths["timestamps"]
    cache_dir = paths["cache"]

    emb_path = cache_dir / f"{video_name}_embeddings.npy"
    ts_path = cache_dir / f"{video_name}_timestamps.npy"
    if not emb_path.exists():
        print(f"[FATAL] missing embedding fixture: {emb_path}", flush=True)
        sys.exit(2)
    if not ts_path.exists():
        print(f"[FATAL] missing timestamp fixture: {ts_path}", flush=True)
        sys.exit(2)

    meta = probe_video_metadata(video_path_obj)
    video_duration_sec = float(meta["duration_sec"])

    # Replace stages 1+2 with cache load.
    print("[1/6] [CACHED] loading embeddings + timestamps from output/cache/...")
    embeddings = np.load(emb_path)
    timestamps = np.load(ts_path)
    print(f"[1/6] embeddings shape={embeddings.shape}  timestamps shape={timestamps.shape}")
    print("[2/6] [CACHED] CLIP step skipped (used Phase 1 fixture)")

    # ── Stage 3: deltas ────────────────────────────────────────────────────
    print("[3/6] Computing embedding deltas...")
    raw_deltas = compute_deltas(embeddings)

    # ── Stage 4: smooth + MAD + (opt) PELT ─────────────────────────────────
    print("[4/6] Filtering and normalizing signal...")
    smoothed = smooth_deltas(raw_deltas)
    scores = mad_normalize(smoothed, window_samples=180)
    changepoints = detect_changepoints(smoothed, penalty=3.0) if use_pelt else None

    # ── Stage 5: peaks → boost → budget → build → merge → enforce ──────────
    print("[5/6] Selecting clips...")
    peak_indices, peak_scores = select_peaks(
        scores, timestamps, height=height, min_gap_sec=min_gap_sec, fps=2.0,
    )
    if use_pelt and changepoints is not None:
        peak_indices, peak_scores = apply_pelt_boost(
            peak_indices, peak_scores, changepoints, window=5, boost=1.2,
        )

    peak_time_to_idx: dict[float, int] = {
        score_index_to_timestamp(int(idx), timestamps): int(idx)
        for idx in peak_indices
    }

    budget_sec = compute_budget_seconds(video_duration_sec)
    padding_sec = compute_padding(budget_sec)
    print(f"[5/6] {len(peak_indices)} peaks; budget={budget_sec:.3f}s "
          f"padding={padding_sec:.3f}s")
    clips = build_clips(peak_indices, peak_scores, timestamps,
                        video_duration_sec, padding_sec)
    merged = merge_clips(clips, gap_threshold_sec=merge_gap_sec)
    final_clips = enforce_budget(merged, budget_sec)
    total = sum(e - s for s, e, _, _ in final_clips)
    print(f"[5/6] {len(final_clips)} final clips; total={total:.3f}s "
          f"({100*total/budget_sec:.1f}% of budget)")

    # ── Stage 6: extract clips, concat reel, write JSON ────────────────────
    print("[6/6] Exporting highlight reel...")
    clip_paths: list[Path] = []
    for i, (start, end, _, _) in enumerate(final_clips):
        out_clip = clips_dir / f"{i:03d}.mp4"
        extract_clip(video_path_obj, out_clip, start, end)
        clip_paths.append(out_clip)
        print(f"[6/6] clip {i+1}/{len(final_clips)}  {start:.3f}..{end:.3f}  "
              f"-> {out_clip.name}")

    reel_path = reels_dir / f"{video_name}_highlight.mp4"
    concat_clips([str(p) for p in clip_paths], str(reel_path), str(clips_dir))

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

    # JSON-04 alignment-invariant defense.
    for clip in manifest["clips"]:
        assert clip["start_sec"] <= clip["peak_timestamp_sec"] <= clip["end_sec"], \
            f"alignment invariant violated: {clip}"

    # D-72 / Pitfall 17 strict three-state validation.
    for clip in manifest["clips"]:
        assert "coincides_with_pelt_changepoint" in clip
        if not use_pelt:
            assert clip["coincides_with_pelt_changepoint"] is None, clip
        else:
            assert isinstance(clip["coincides_with_pelt_changepoint"], bool), clip

    json_path = timestamps_dir / f"{video_name}.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Done. Reel: {reel_path}")
    print(f"      JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache-bypass Phase 5 harness")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--pelt", action="store_true", default=False)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--min-gap-sec", type=float, default=15.0)
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    args = parser.parse_args()
    t0 = time.monotonic()
    run_from_cache(
        args.video,
        use_pelt=args.pelt,
        height=args.height,
        min_gap_sec=args.min_gap_sec,
        merge_gap_sec=args.merge_gap_sec,
    )
    print(f"Total wall-clock: {time.monotonic() - t0:.2f}s")
