"""Phase 3: peak detection, adaptive padding, merge, budget enforcement.

Module API (consumed by Phase 5's pipeline.run()):
    select_peaks(scores, timestamps, height=1.5, min_gap_sec=15.0, fps=2.0)
        -> (peak_indices, peak_scores)        # sorted by descending score
    apply_pelt_boost(peak_indices, peak_scores, changepoints, window=5, boost=1.2)
        -> (reordered_indices, boosted_scores)  # opt-in, called only when --pelt
    compute_budget_seconds(video_duration_sec) -> float
    compute_padding(budget_sec, max_padding=8.0, min_padding=3.0) -> float
    build_clips(peak_indices, peak_scores, timestamps, video_duration_sec, padding_sec)
        -> list[(start, end, score, peak_time)]    # chronological 4-tuples (D-38)
    merge_clips(clips, gap_threshold_sec=3.0)
        -> list[(start, end, score, peak_time)]    # peak_time follows higher score (Pitfall 19)
    enforce_budget(merged_clips, budget_sec)
        -> list[(start, end, score, peak_time)]    # final chronological list

Verification harness (spec §0.5) lives in `if __name__ == "__main__":` per D-49.
Run: `python clip_selection.py [<video_name>] [--pelt] [--height H] [--min-gap-sec G] [--merge-gap-sec M]`
to verify against cached Phase 1+2 fixtures at `output/cache/<video>_{scores,timestamps}.npy`.

Phase 3 owns 3 pitfalls (research/PITFALLS.md):
  18 (partial-clip on peak — enforce_budget centers on peak_time when remainder >= 3s),
  19 (peak_time through merge — 4-tuple threading + start <= peak_time <= end runtime guard),
  20 (find_peaks distance unit — int(min_gap_sec * fps) inlined + post-call assertion).
Each is covered by a runtime assertion and an inline test in the __main__ harness.

Cross-cutting alignment invariant: ``score[k]`` aligns with ``timestamps`` at offset
``k+1`` (Phase 1+2 STATE.md Cross-Cutting Invariants §1). Phase 3 IS the conversion
seam — `build_clips` is the FIRST function in the codebase to legitimately pair an
index with a timestamp, and it does so via the imported `score_index_to_timestamp`
helper from signal_processing (D-24, D-45). Inline ``timestamps[k + 1]`` style
arithmetic is grep-banned in this file (D-45 / Pitfall 8).

Two locked deviations from spec §6 verbatim (documented inline at each function):
  - D-38 / D-46: 4-tuple (start, end, score, peak_time) instead of 3-tuple
                 (start, end, score) — peak_time threading through merge so JSON
                 consumers can populate peak_timestamp_sec with the alignment
                 invariant guarantee ``start <= peak_timestamp_sec <= end``.
  - D-47: enforce_budget partial clip is CENTERED ON peak_time when remainder >= 3s,
          not truncated at (clip.start, clip.start + remainder) — Pitfall 18 fix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from signal_processing import score_index_to_timestamp


def select_peaks(
    scores: np.ndarray,
    timestamps: np.ndarray,
    height: float = 1.5,
    min_gap_sec: float = 15.0,
    fps: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks in MAD-normalized scores; sort by descending score.

    DEVIATION-FREE from spec §6: matches the spec snippet verbatim with one
    defensive addition — a post-call assertion that the find_peaks `distance`
    contract was honored (Pitfall 20).

    Args:
        scores: (N-1,) float64 MAD-normalized score array (output of mad_normalize).
        timestamps: (N,) float64 frame timestamps. Used only for the §0.5 print in
            __main__; the function body itself stays in score-index space (D-40).
            Conversion to seconds is the responsibility of build_clips, via the
            imported `score_index_to_timestamp` helper.
        height: minimum MAD score to qualify as a peak (default 1.5 per spec §6).
        min_gap_sec: minimum seconds between selected peaks (default 15.0 per spec §6).
        fps: sampling rate (default 2.0 — locked at 2 fps per Phase 1 D-04).

    Returns:
        (peak_indices, peak_scores) — both np.ndarray, sorted by DESCENDING score.
        peak_indices.dtype is integer; peak_scores.dtype is float64.

    Pitfall 20: scipy.signal.find_peaks `distance` is in SAMPLES, not seconds.
    The unit conversion is inlined at the call site so a reader sees `int(min_gap_sec * fps)`.
    """
    min_gap_samples = int(min_gap_sec * fps)
    peaks, _ = find_peaks(scores, height=height, distance=min_gap_samples)
    # Pitfall 20 post-call assertion: find_peaks honors `distance` in samples.
    # `np.diff(np.sort(peaks))` is the gap between adjacent peaks in index space.
    # Guard against scipy version drift; passes for scipy>=1.16.2.
    if len(peaks) >= 2:
        assert int(np.diff(np.sort(peaks)).min()) >= min_gap_samples, (
            f"find_peaks violated distance contract: "
            f"min gap {int(np.diff(np.sort(peaks)).min())} < {min_gap_samples} samples "
            "(Pitfall 20)"
        )
    # Sort by descending score per spec §6 / SELP-02.
    peak_scores = scores[peaks]
    sorted_idx = np.argsort(peak_scores)[::-1]
    return peaks[sorted_idx], peak_scores[sorted_idx]


def apply_pelt_boost(
    peak_indices: np.ndarray,
    peak_scores: np.ndarray,
    changepoints: list[int],
    window: int = 5,
    boost: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply peak scores by `boost` when within ±`window` samples of a changepoint.

    DEVIATION-FREE from spec §6: matches the spec snippet verbatim. Pure
    index-space — takes no timestamps. When --pelt is OFF, Phase 5's pipeline.run()
    does NOT call this function (D-42 / cross-cutting --pelt orthogonality).

    Args:
        peak_indices: (K,) int ndarray (typically the first return of select_peaks).
        peak_scores: (K,) float ndarray — corresponding scores.
        changepoints: list[int] of indices (output of signal_processing.detect_changepoints).
        window: sample radius for "near a changepoint" (default 5 per spec §5).
        boost: multiplicative factor (default 1.2 per spec §5).

    Returns:
        (reordered_indices, boosted_scores) — both np.ndarray, sorted by descending
        boosted score. Original peak_scores is NOT mutated (we operate on a copy).
    """
    boosted_scores = peak_scores.copy()
    for i, idx in enumerate(peak_indices):
        if any(abs(int(idx) - cp) <= window for cp in changepoints):
            boosted_scores[i] *= boost
    sorted_idx = np.argsort(boosted_scores)[::-1]
    return peak_indices[sorted_idx], boosted_scores[sorted_idx]


def compute_budget_seconds(video_duration_sec: float) -> float:
    """Reel duration budget: 1 minute of reel per 30 minutes of video (spec §6)."""
    return (video_duration_sec / 1800.0) * 60.0


def compute_padding(
    budget_sec: float,
    max_padding: float = 8.0,
    min_padding: float = 3.0,
) -> float:
    """Adaptive per-side padding: 15% of budget, clamped to [min_padding, max_padding]
    seconds. Prevents short videos from having clips longer than the budget itself
    (spec §6: "For a 3-minute video → 3.0s each side; for a 30-minute video → 8.0s").
    """
    return max(min_padding, min(max_padding, budget_sec * 0.15))


def build_clips(
    peak_indices: np.ndarray,
    peak_scores: np.ndarray,
    timestamps: np.ndarray,
    video_duration_sec: float,
    padding_sec: float,
) -> list[tuple[float, float, float, float]]:
    """Build candidate (start, end, score, peak_time) 4-tuples around each peak.

    THIS IS THE SINGLE INDEX->SECONDS CONVERSION SEAM IN THE PIPELINE (D-45).
    Uses `score_index_to_timestamp(idx, timestamps)` from signal_processing —
    which returns the appropriate frame timestamp per the alignment invariant
    (Pitfall 8). Inline timestamp arithmetic is FORBIDDEN here (grep-verified).

    Args:
        peak_indices: (K,) int ndarray of score-array indices.
        peak_scores: (K,) float ndarray of corresponding (possibly boosted) scores.
        timestamps: (N,) float64 frame timestamps from Phase 1 fixture.
        video_duration_sec: from `utils.probe_video_metadata(...)["duration_sec"]`.
        padding_sec: from `compute_padding(budget)`.

    Returns:
        list[tuple[start, end, score, peak_time]] sorted CHRONOLOGICALLY by `start`.
        All elements are Python floats (clean for JSON serialization in Phase 5).

    Pitfall 19 setup: peak_time is the 4th tuple element from creation onward,
    threaded through merge_clips → enforce_budget so the alignment invariant
    ``start <= peak_time <= end`` holds for every clip in the final list.
    """
    clips: list[tuple[float, float, float, float]] = []
    for idx, score in zip(peak_indices, peak_scores):
        peak_time = score_index_to_timestamp(int(idx), timestamps)  # D-45 / Pitfall 8
        start = max(0.0, peak_time - padding_sec)
        end = min(video_duration_sec, peak_time + padding_sec)
        clips.append((float(start), float(end), float(score), float(peak_time)))
    clips.sort(key=lambda c: c[0])  # chronological order
    return clips


def merge_clips(
    clips: list[tuple[float, float, float, float]],
    gap_threshold_sec: float = 3.0,
) -> list[tuple[float, float, float, float]]:
    """Merge chronologically-adjacent clips within `gap_threshold_sec`; keep the
    higher score AND its associated `peak_time` (Pitfall 19 / D-46). On score tie,
    keep the earlier peak_time (deterministic).

    DEVIATION from spec §6 snippet: spec uses 3-tuples `(start, end, score)` and
    discards `peak_time` on merge. Phase 3 uses 4-tuples (D-38) so JSON consumers
    in Phase 5 can populate `peak_timestamp_sec` with the alignment invariant
    ``start <= peak_timestamp_sec <= end`` guaranteed.

    Args:
        clips: list[(start, end, score, peak_time)] sorted chronologically.
        gap_threshold_sec: merge if `start <= prev.end + gap_threshold_sec`.
            Default 3.0 per spec §6 (also CLI flag --merge-gap-sec in Phase 5).

    Returns:
        list[(start, end, score, peak_time)] sorted chronologically. May be shorter
        than input. For each output clip, ``start <= peak_time <= end`` is HARD-ASSERTED
        before return (Pitfall 19 runtime guard).
    """
    if not clips:
        return []
    merged: list[list[float]] = [list(clips[0])]
    for start, end, score, peak_time in clips[1:]:
        prev = merged[-1]  # [prev_start, prev_end, prev_score, prev_peak_time]
        if start <= prev[1] + gap_threshold_sec:
            # Merge: extend end to max(prev.end, this.end). Score & peak_time follow
            # the higher-scoring source (Pitfall 19). Tie-break: keep earlier peak_time.
            prev[1] = max(prev[1], end)
            if score > prev[2]:
                prev[2] = score
                prev[3] = peak_time
            elif score == prev[2] and peak_time < prev[3]:
                prev[3] = peak_time  # tie-break: earlier peak_time wins (deterministic)
            # else: keep prev's score and peak_time
        else:
            merged.append([start, end, score, peak_time])
    out = [tuple(c) for c in merged]
    # Pitfall 19 runtime guard: alignment invariant must hold post-merge.
    for c in out:
        assert c[0] <= c[3] <= c[1], (
            f"merge_clips broke alignment invariant: start={c[0]} peak_time={c[3]} "
            f"end={c[1]} (Pitfall 19)"
        )
    return out


def enforce_budget(
    merged_clips: list[tuple[float, float, float, float]],
    budget_sec: float,
) -> list[tuple[float, float, float, float]]:
    """Greedy budget enforcement: select by descending score until budget is reached;
    if the next clip doesn't fit but remainder >= 3.0s, append a partial clip CENTERED
    ON `peak_time` (Pitfall 18 / D-47).

    DEVIATION from spec §6 snippet: spec's partial-clip logic truncates at
    `(clip.start, clip.start + remaining)` — which can EXCLUDE the actual peak when
    padding is large. Phase 3 centers the partial clip on peak_time so the
    highest-confidence moment is always inside the final reel (Pitfall 18 fix).

    Args:
        merged_clips: list[(start, end, score, peak_time)] post-merge.
        budget_sec: from `compute_budget_seconds(video_duration_sec)`.

    Returns:
        list[(start, end, score, peak_time)] sorted CHRONOLOGICALLY, with
        ``sum(end - start) <= budget_sec + 1e-6`` HARD-ASSERTED and no two clips
        overlapping (SELB-06).

    Numerical-safety constants (D-47 / Claude's Discretion):
        1e-6: budget total assertion tolerance.
        0.95: peak-too-close-to-edge fallback ratio — if the centered partial
              window comes back narrower than 95% of `remainder`, fall back to
              `(clip.start, clip.start + remainder)`.
    """
    # Stable sort by descending score (Python's sort is stable; D-47 / Claude's Discretion).
    by_score = sorted(merged_clips, key=lambda c: c[2], reverse=True)
    selected: list[tuple[float, float, float, float]] = []
    total = 0.0
    for clip in by_score:
        start, end, score, peak_time = clip
        duration = end - start
        if total + duration <= budget_sec:
            selected.append(clip)
            total += duration
        else:
            remainder = budget_sec - total
            if remainder >= 3.0:
                # Partial clip CENTERED ON peak_time (Pitfall 18).
                half = remainder / 2.0
                partial_start = max(start, peak_time - half)
                partial_end = min(end, partial_start + remainder)
                # If the centered window is squeezed (peak near clip edge), fall back
                # to truncating at clip.start (D-47 fallback ratio 0.95).
                if (partial_end - partial_start) < remainder * 0.95:
                    partial_start = start
                    partial_end = min(end, start + remainder)
                selected.append((
                    float(partial_start), float(partial_end),
                    float(score), float(peak_time),
                ))
                total += (partial_end - partial_start)
            break  # spec §6 — one partial then stop
    # Re-sort chronologically for export (SELB-05 final step).
    selected.sort(key=lambda c: c[0])
    # SELB-05 / SELB-06 hard assertions.
    assert total <= budget_sec + 1e-6, (
        f"enforce_budget overshot: total={total:.6f} > budget={budget_sec:.6f}"
    )
    for i in range(len(selected) - 1):
        assert selected[i][1] <= selected[i + 1][0], (
            f"enforce_budget output overlaps: clip[{i}]={selected[i]} "
            f"clip[{i + 1}]={selected[i + 1]} (SELB-06)"
        )
    # Pitfall 18 / Pitfall 19 invariant survives partial-clip rewriting:
    for c in selected:
        assert c[0] <= c[3] <= c[1], (
            f"enforce_budget broke alignment invariant: start={c[0]} peak_time={c[3]} "
            f"end={c[1]} (Pitfalls 18 + 19)"
        )
    return selected
