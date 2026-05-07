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
                # Partial clip CENTERED ON peak_time (Pitfall 18 / D-47).
                # Pick a window of width `remainder` centered on peak_time, then
                # shift it (preserving width) to stay inside [start, end].
                # This guarantees `partial_start <= peak_time <= partial_end`
                # because peak_time ∈ [start, end] is asserted by merge_clips
                # (Pitfall 19 / D-46), and shifting only happens to anchor a
                # boundary at start or end.
                half = remainder / 2.0
                partial_start = peak_time - half
                partial_end = peak_time + half
                # Left clamp: if window pokes past the start edge, shift right.
                if partial_start < start:
                    partial_end = min(end, partial_end + (start - partial_start))
                    partial_start = start
                # Right clamp: if window pokes past the end edge, shift left.
                if partial_end > end:
                    partial_start = max(start, partial_start - (partial_end - end))
                    partial_end = end
                # NOTE: after both clamps, partial may be NARROWER than `remainder`
                # iff the source clip's duration was already < remainder. In that
                # case partial == [start, end] and peak_time is inside by construction.
                # We previously had a fallback (D-47 0.95 ratio) that anchored to
                # clip.start when the centered window was squeezed; that fallback
                # discarded peak_time and broke the alignment invariant when peak
                # was close to clip's right edge with a large budget remainder.
                # Removed in favor of the shift-preserving logic above.
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


if __name__ == "__main__":
    # Spec §0.5 verification harness for Phase 3 (D-49). Runs as:
    #   python clip_selection.py [<video_name>] [--save-fixture | --no-save-fixture]
    #     [--pelt] [--height H] [--min-gap-sec G] [--merge-gap-sec M]
    # Mirrors signal_processing.py's __main__ pattern verbatim per D-49 / Phase 1 D-16.
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 §0.5 verification harness for clip_selection.py "
            "(D-49: inline __main__ — no separate test scaffolding per "
            "Phase 1 D-16 / Phase 2 D-34)."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help=(
            "Video name stem (e.g. 'justin_timberlake'). If omitted, picks the "
            "most-recent *_scores.npy under output/cache/ by mtime."
        ),
    )
    parser.add_argument(
        "--save-fixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write output/cache/{video}_final_clips.json (D-50 — Phase 4 dev "
            "fixture). Default ON; pass --no-save-fixture to skip."
        ),
    )
    parser.add_argument(
        "--pelt",
        action="store_true",
        default=False,
        help=(
            "Apply PELT score boost. Loads output/cache/{video}_changepoints.npy "
            "(produced by `python signal_processing.py --pelt`). Off by default."
        ),
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.5,
        help="MAD score threshold for find_peaks (spec §6 default 1.5).",
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=15.0,
        help="Minimum seconds between peaks (spec §6 default 15.0).",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=3.0,
        help="Maximum gap between adjacent clips to merge (spec §6 default 3.0).",
    )
    args = parser.parse_args()

    cache = Path("output/cache")

    # ── Step 0: resolve video stem (smart default) ─────────────────────────────
    if args.video is None:
        candidates = sorted(
            cache.glob("*_scores.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                "[FATAL] no *_scores.npy under output/cache/. "
                "Run `python signal_processing.py <video>` first.",
                file=sys.stderr,
            )
            sys.exit(2)
        video_name = candidates[0].stem.removesuffix("_scores")
        print(f"[auto] using fixture: {video_name}")
    else:
        video_name = args.video

    # ── Step 1: load fixtures ──────────────────────────────────────────────────
    scores_path = cache / f"{video_name}_scores.npy"
    ts_path = cache / f"{video_name}_timestamps.npy"
    if not scores_path.exists() or not ts_path.exists():
        print(
            f"[FATAL] missing fixture: {scores_path} or {ts_path}",
            file=sys.stderr,
        )
        sys.exit(2)
    scores = np.load(scores_path)        # (N-1,) float64
    timestamps = np.load(ts_path)        # (N,) float64
    print(
        f"[fixture] loaded {video_name}: "
        f"scores={scores.shape} timestamps={timestamps.shape}"
    )

    # Probe duration (single source of truth — utils.probe_video_metadata,
    # NOT timestamps[-1]).
    from utils import probe_video_metadata
    video_path = Path("videos") / f"{video_name}.mp4"
    if not video_path.exists():
        print(f"[FATAL] video file not found: {video_path}", file=sys.stderr)
        sys.exit(2)
    meta = probe_video_metadata(video_path)
    video_duration_sec = float(meta["duration_sec"])
    print(
        f"[meta] duration={video_duration_sec:.3f}s codec={meta['codec']} "
        f"is_vfr={meta['is_vfr']}"
    )

    # Optional --pelt: load changepoints fixture
    changepoints: list[int] | None = None
    if args.pelt:
        cp_path = cache / f"{video_name}_changepoints.npy"
        if not cp_path.exists():
            print(
                f"[FATAL] --pelt set but missing {cp_path}. "
                f"Run `python signal_processing.py {video_name} --pelt` first.",
                file=sys.stderr,
            )
            sys.exit(2)
        changepoints = np.load(cp_path).astype(int).tolist()
        print(f"[pelt] loaded {len(changepoints)} changepoints")

    # ── Step 2: select_peaks + §0.5 print after peaks (D-41 / SELP-04) ─────────
    peak_indices, peak_scores = select_peaks(
        scores, timestamps,
        height=args.height, min_gap_sec=args.min_gap_sec, fps=2.0,
    )
    # Pitfall 20 secondary echo (function asserts internally; print for operator).
    if len(peak_indices) >= 2:
        min_gap_samples_observed = int(np.diff(np.sort(peak_indices)).min())
        assert min_gap_samples_observed >= int(args.min_gap_sec * 2.0), (
            f"[peaks] min gap {min_gap_samples_observed} < "
            f"{int(args.min_gap_sec * 2.0)} samples (Pitfall 20)"
        )
    print(f"[peaks] count={len(peak_indices)}")
    preview_n = min(10, len(peak_indices))
    for i in range(preview_n):
        idx = int(peak_indices[i])
        pt = score_index_to_timestamp(idx, timestamps)
        print(
            f"[peaks]   #{i}: idx={idx} ts={pt:.3f}s "
            f"score={peak_scores[i]:.4f}"
        )

    # ── Step 3: optional PELT boost (D-42 / SELP-03) ───────────────────────────
    if args.pelt and changepoints is not None:
        peak_indices, peak_scores = apply_pelt_boost(
            peak_indices, peak_scores, changepoints, window=5, boost=1.2,
        )
        print(f"[pelt-boost] re-sorted; top score now {peak_scores[0]:.4f}")

    # ── Step 4: compute_budget_seconds + compute_padding ───────────────────────
    budget_sec = compute_budget_seconds(video_duration_sec)
    padding_sec = compute_padding(budget_sec)
    print(
        f"[budget] duration={video_duration_sec:.3f}s -> "
        f"budget={budget_sec:.3f}s padding={padding_sec:.3f}s"
    )

    # ── Step 5: build_clips ────────────────────────────────────────────────────
    clips = build_clips(
        peak_indices, peak_scores, timestamps,
        video_duration_sec, padding_sec,
    )
    for c in clips:
        assert c[0] <= c[3] <= c[1], f"[build_clips] alignment broken: {c}"
    print(f"[build_clips] {len(clips)} candidate clips (chronological)")

    # ── Step 6: merge_clips (Pitfall 19 hard-asserted inside the function) ────
    merged = merge_clips(clips, gap_threshold_sec=args.merge_gap_sec)
    print(
        f"[merge] {len(clips)} -> {len(merged)} clips after "
        f"gap_threshold={args.merge_gap_sec}s"
    )
    for c in merged:
        assert c[0] <= c[3] <= c[1], f"[merge] alignment broken: {c}"

    # ── Step 7: enforce_budget + §0.5 final clip print (D-48 / SELB-06) ───────
    final = enforce_budget(merged, budget_sec)
    total = sum(c[1] - c[0] for c in final)
    pct = (total / budget_sec * 100.0) if budget_sec > 0 else 0.0
    print(f"[budget] total={total:.3f}s / budget={budget_sec:.3f}s ({pct:.1f}%)")
    for i, (s, e, sc, pt) in enumerate(final):
        dur = e - s
        print(
            f"[clip-{i}] start={s:.3f} end={e:.3f} dur={dur:.3f} "
            f"score={sc:.4f} peak_time={pt:.3f}"
        )
    # Echo asserts (functions already enforce; surface for operator).
    assert total <= budget_sec + 1e-6, f"[budget] over: {total} > {budget_sec}"
    for c in final:
        assert c[0] <= c[3] <= c[1], f"[final] alignment broken: {c}"
    for i in range(len(final) - 1):
        assert final[i][1] <= final[i + 1][0], (
            f"[final] overlap: {final[i]} vs {final[i + 1]}"
        )

    # ── Step 8: synthetic embedded test (D-49) ─────────────────────────────────
    # Catches off-by-ones in merge / budget logic that pure-fixture tests miss.
    synth_scores = np.full(100, 0.3)
    for idx, h in [(20, 4.0), (40, 5.0), (60, 3.0), (80, 6.0)]:
        synth_scores[idx] = h
    synth_ts = np.arange(101) * 0.5  # 0.0..50.0
    synth_pi, synth_ps = select_peaks(
        synth_scores, synth_ts, height=1.5, min_gap_sec=5.0, fps=2.0,
    )  # min_gap_samples = 10 → all 4 peaks survive (gaps are 20)
    assert len(synth_pi) == 4, f"[synth] peaks: {len(synth_pi)}"
    synth_clips = build_clips(
        synth_pi, synth_ps, synth_ts,
        video_duration_sec=50.0, padding_sec=3.0,
    )
    assert len(synth_clips) == 4
    # peak idx 20 → ts at offset 21 = 10.5; clip = (7.5, 13.5, 4.0, 10.5)
    synth_chrono = sorted(synth_clips, key=lambda c: c[0])
    expected_first = (7.5, 13.5, 4.0, 10.5)
    assert synth_chrono[0] == expected_first, (
        f"[synth] first clip: {synth_chrono[0]} != {expected_first}"
    )
    synth_merged = merge_clips(synth_clips, gap_threshold_sec=3.0)
    # 6s-wide clips with 7s gaps → no merges expected; alignment must hold.
    assert len(synth_merged) >= 1
    for c in synth_merged:
        assert c[0] <= c[3] <= c[1]
    synth_final = enforce_budget(synth_merged, budget_sec=10.0)
    synth_total = sum(c[1] - c[0] for c in synth_final)
    assert synth_total <= 10.0 + 1e-6
    print("[synthetic] PASS  (peaks, build, merge, budget all chained correctly)")

    # ── Step 9: write fixture (D-50) ───────────────────────────────────────────
    if args.save_fixture:
        from utils import ensure_output_dirs
        paths = ensure_output_dirs(video_name)
        cache_dir = paths["cache"]
        fixture_path = cache_dir / f"{video_name}_final_clips.json"
        payload = [
            {"start_sec": s, "end_sec": e, "score": sc, "peak_time": pt}
            for s, e, sc, pt in final
        ]
        with open(fixture_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[fixture] wrote {fixture_path}")

    # ── Step 10: final banner ──────────────────────────────────────────────────
    print("Phase 3 §0.5 verification: PASS")
