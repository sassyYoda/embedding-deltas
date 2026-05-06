"""Phase 4: ffmpeg/ffprobe subprocess orchestration for clip extraction + concatenation.

Module API (consumed by Phase 5's pipeline.run()):
    extract_clip(input_path, output_path, start_sec, end_sec) -> None
        Stream-copy cut via `ffmpeg -y -ss <start> -to <end> -i <input> -c copy <output>`.
        `-ss` BEFORE `-i` is the FAST input-seek path (D-52 / Pitfall 11).
    validate_clips_for_concat(clip_paths) -> tuple[bool, str]
        ffprobe pre-flight: codec_name + width + height + r_frame_rate + time_base
        consistency across clips, plus all-or-none audio-stream presence (D-57 / Pitfall 12).
        Returns (True, "") if compatible; (False, "<reason>") otherwise.
    concat_clips(clip_paths, output_path, temp_dir) -> None
        Validates first; on PASS uses concat demuxer (`-f concat -safe 0 -c copy`); on
        FAIL falls back to concat-filter re-encode (`libx264 -crf 18 / aac 192k`).
        Logs `[concat] PATH=demuxer ...` or `[concat] FALLBACK: ...` for operator
        visibility (D-55).

Verification harness (spec §0.5) lives in `if __name__ == "__main__":` per D-58.
Run: `python export.py [<video_name>]` to drive the three functions against the Phase 3
fixture at `output/cache/<video>_final_clips.json` and produce the Phase 4 deliverable
artifact at `output/reels/<video>_highlight.mp4`.

Phase 4 owns 3 pitfalls (research/PITFALLS.md):
  11 (`-ss` placement before `-i` for stream-copy fast seek; ~1s imprecision acceptable
      per spec §10),
  12 (concat demuxer fragility — pre-validate codec/timebase, fall back to concat-filter
      re-encode on mismatch),
  13 (concat manifest path safety — `os.path.abspath` + `-safe 0` + 4-char single-quote
      escape `'` -> `'\''`).

Import surface is stdlib + utils only per D-60 — NO numpy, torch, cv2, open_clip,
ruptures. Phase 4 is pure subprocess orchestration over ffmpeg/ffprobe.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def extract_clip(
    input_path: str | Path,
    output_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Stream-copy a single clip from `input_path` between `start_sec` and `end_sec`.

    Implements D-52, D-53, D-54:
      - D-52 / Pitfall 11: `-ss` BEFORE `-i` (FAST input-seek path with `-c copy`).
      - D-53: post-call `os.path.getsize > 0` guard against silent zero-byte output.
      - D-54: on `subprocess.CalledProcessError`, re-raise with decoded stderr in the
              message so the caller sees ffmpeg's diagnostic.

    Args:
        input_path: source video (str or pathlib.Path).
        output_path: target .mp4 path (str or pathlib.Path); parent dir must exist.
        start_sec: clip start in seconds (float). Honored to nearest preceding keyframe;
                   per spec §10 ~1s earlier-than-requested is acceptable.
        end_sec:   clip end in seconds (float).

    Raises:
        RuntimeError: on ffmpeg subprocess failure (D-54) OR zero-byte output (D-53).
    """
    in_str = os.fspath(input_path)
    out_str = os.fspath(output_path)
    cmd = [
        "ffmpeg", "-y",
        # Pitfall 11: -ss BEFORE -i is the FAST stream-copy seek path; ~1s imprecision acceptable per spec §10.
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", in_str,
        "-c", "copy",
        out_str,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # D-54: surface ffmpeg's stderr in the raised message — never swallow.
        raise RuntimeError(
            f"ffmpeg extract_clip failed for {out_str}: "
            f"{e.stderr.decode('utf-8', errors='replace')}"
        ) from e
    # D-53: ffmpeg can occasionally produce a zero-byte file without raising
    # (e.g., start_sec past EOF on some builds). Guard explicitly.
    if os.path.getsize(out_str) <= 0:
        raise RuntimeError(f"clip extraction produced empty file: {out_str}")


def validate_clips_for_concat(
    clip_paths: list[str | Path],
) -> tuple[bool, str]:
    """ffprobe pre-flight check that all clips are concat-demuxer-compatible (D-57).

    Implements D-57 / Pitfall 12: probes each clip's video stream for codec_name,
    width, height, r_frame_rate, time_base. All clips must agree on every field.
    Audio: probes presence + codec_name; mixed presence (some have audio, some don't)
    fails validation; when all have audio, codec_name must match.

    Args:
        clip_paths: list of clip .mp4 paths (str or pathlib.Path). Empty list returns
                    (True, "") — caller handles the empty case separately.

    Returns:
        (True, "")               — all clips compatible; safe to use concat demuxer.
        (False, "<reason>")      — incompatible; caller should use concat-filter fallback.
                                   Reason includes the specific field + which clips disagreed.
    """
    if not clip_paths:
        return (True, "")

    video_metas: list[dict] = []
    audio_metas: list[dict | None] = []  # None when no audio stream

    for p in clip_paths:
        p_str = os.fspath(p)
        # Probe video stream (required).
        video_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,time_base",
            "-of", "json",
            p_str,
        ]
        try:
            v_proc = subprocess.run(video_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            return (False, f"ffprobe failed for {p_str}: "
                           f"{e.stderr.decode('utf-8', errors='replace').strip() if isinstance(e.stderr, bytes) else (e.stderr or '').strip()}")
        v_streams = json.loads(v_proc.stdout).get("streams", [])
        if not v_streams:
            return (False, f"no video stream in {p_str}")
        video_metas.append(v_streams[0])

        # Probe audio stream presence (optional).
        audio_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "json",
            p_str,
        ]
        try:
            a_proc = subprocess.run(audio_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            return (False, f"ffprobe (audio) failed for {p_str}: "
                           f"{e.stderr.decode('utf-8', errors='replace').strip() if isinstance(e.stderr, bytes) else (e.stderr or '').strip()}")
        a_streams = json.loads(a_proc.stdout).get("streams", [])
        audio_metas.append(a_streams[0] if a_streams else None)

    # Compare every video field against clip 0 (D-57).
    first_v = video_metas[0]
    for i in range(1, len(video_metas)):
        this_v = video_metas[i]
        for field in ("codec_name", "width", "height", "r_frame_rate", "time_base"):
            if first_v.get(field) != this_v.get(field):
                return (False,
                        f"video {field} mismatch: clip 0 {first_v.get(field)!r} "
                        f"vs clip {i} {this_v.get(field)!r}")

    # Audio presence + codec_name consistency.
    first_has_audio = audio_metas[0] is not None
    for i in range(1, len(audio_metas)):
        this_has_audio = audio_metas[i] is not None
        if this_has_audio != first_has_audio:
            return (False, "mixed audio presence: clips disagree on audio stream existence")
    if first_has_audio:
        first_a_codec = audio_metas[0].get("codec_name")  # type: ignore[union-attr]
        for i in range(1, len(audio_metas)):
            this_a_codec = audio_metas[i].get("codec_name")  # type: ignore[union-attr]
            if first_a_codec != this_a_codec:
                return (False, f"audio codec mismatch: clip 0 {first_a_codec!r} "
                               f"vs clip {i} {this_a_codec!r}")

    return (True, "")


def concat_clips(
    clip_paths: list[str | Path],
    output_path: str | Path,
    temp_dir: str | Path,
) -> None:
    """Concatenate `clip_paths` into a single reel at `output_path` (D-55, D-56).

    Pre-flight: runs `validate_clips_for_concat` (Pitfall 12). On PASS, uses the
    concat demuxer (lossless, fast); on FAIL, falls back to the concat filter
    (re-encodes via libx264 + aac — slower but reliable). Logs the chosen path to
    stdout for operator visibility (D-55).

    Manifest path safety (D-56 / Pitfall 13):
      - `os.path.abspath` for every clip path,
      - `-safe 0` on the demuxer command,
      - single-quote escape `'` -> `'\\''` (the 4-char ffmpeg literal).

    Args:
        clip_paths: list of clip .mp4 paths (str or pathlib.Path). Must be non-empty.
        output_path: reel target .mp4 path (str or pathlib.Path).
        temp_dir: directory for the concat manifest (caller-managed; e.g. the
                  per-video clips/ subdir from utils.ensure_output_dirs).

    Raises:
        ValueError: if `clip_paths` is empty.
        RuntimeError: on ffmpeg subprocess failure (D-54) OR zero-byte reel output.
    """
    if not clip_paths:
        raise ValueError("concat_clips requires at least one clip path")

    out_str = os.fspath(output_path)
    tmp_str = os.fspath(temp_dir)

    # Pitfall 12: pre-validate; choose demuxer (compatible) or filter (fallback).
    ok, reason = validate_clips_for_concat(clip_paths)

    if ok:
        # D-55 visibility: log which path is taken so operators can spot regressions.
        print("[concat] PATH=demuxer (reason: validate_clips_for_concat OK)")

        manifest_path = os.path.join(tmp_str, "concat_manifest.txt")
        with open(manifest_path, "w") as f:
            for path in clip_paths:
                abs_path = os.path.abspath(os.fspath(path))
                # Pitfall 13: concat manifest uses single-quoted paths; '\''  is the 4-char single-quote escape ffmpeg expects.
                escaped = abs_path.replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", manifest_path,
            "-c", "copy",
            out_str,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # D-54: surface ffmpeg's stderr.
            raise RuntimeError(
                f"ffmpeg concat (demuxer) failed: "
                f"{e.stderr.decode('utf-8', errors='replace')}"
            ) from e
    else:
        # D-55 visibility: fallback path with the specific validation reason.
        print(f"[concat] FALLBACK: re-encoding via concat filter (reason: {reason})")

        inputs: list[str] = []
        for path in clip_paths:
            inputs.extend(["-i", os.path.abspath(os.fspath(path))])
        n = len(clip_paths)
        filter_str = (
            "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
            + f"concat=n={n}:v=1:a=1[outv][outa]"
        )
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_str,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg concat (fallback re-encode) failed: "
                f"{e.stderr.decode('utf-8', errors='replace')}"
            ) from e

    # D-53: post-call zero-byte guard for the reel as well.
    if os.path.getsize(out_str) <= 0:
        raise RuntimeError(f"concat produced empty file: {out_str}")
