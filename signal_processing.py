"""Phase 2: cosine deltas, median smoothing, rolling MAD, opt-in PELT.

Module API (consumed by Phase 3's clip_selection and Phase 5's pipeline.run()):
    compute_deltas(embeddings) -> (N-1,) float64
    smooth_deltas(raw, kernel_size=5) -> (N-1,) float64       # mode='reflect' (deviation from spec §5)
    mad_normalize(smoothed, window_samples=180) -> (N-1,) float64  # MAD floor 1e-3 (deviation from spec §5)
    detect_changepoints(smoothed, penalty=3.0) -> list[int]   # LAZY-IMPORTS ruptures
    score_index_to_timestamp(i, timestamps) -> float          # the ONLY index→seconds site (D-24)

Verification harness (spec §0.5) lives in `if __name__ == "__main__":` per D-34 / Phase 1 D-16.
Run: `python signal_processing.py [<video_name>]` to verify the module against the cached
Phase 1 fixture at `output/cache/<video_name>_{embeddings,timestamps}.npy`.

Phase 2 owns 3 pitfalls (research/PITFALLS.md): 8 (off-by-one alignment — synthetic test in
__main__), 9 (medfilt edge effects — switched to ndimage.median_filter mode='reflect'),
10 (MAD-zero — floor raised to 1e-3). Each is covered by a runtime assertion in the harness.

Two locked deviations from spec §5 verbatim (documented inline at each function):
  - D-27: scipy.ndimage.median_filter(mode='reflect') instead of the spec's
          scipy.signal medfilt (zero-padding edge effects → phantom dips at
          boundaries — Pitfall 9).
  - D-29: MAD floor 1e-3 (raised from the spec's much-looser float-precision guard);
          static-footage windows produce MAD in the [1e-7, 1e-3] range which would
          round-trip to ceiling-pegged scores under the looser guard (Pitfall 10).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


def compute_deltas(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive embeddings.

    Embeddings are already L2-normalized (Phase 1 D-13 hard-asserts this), so cosine
    similarity reduces to a dot product. Returns ``1 - clip(dot, -1, 1)``.

    Args:
        embeddings: (N, 768) float32 L2-normalized array. N must be >= 2.

    Returns:
        (N-1,) float64 array. ``deltas[i]`` is the change ARRIVING AT frame i+1
        (spec §4 closing note). This is INDEX SPACE — no timestamps involved.
        Use ``score_index_to_timestamp(i, ts)`` for the conversion (D-23, D-24).

    Pitfall: callers must NOT inline ``timestamps[i]`` for delta i — the alignment
    is ``timestamps[i+1]`` (Pitfall 8 / research/PITFALLS.md §8).
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(f"embeddings must be (N, 768); got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError(
            f"need at least 2 embeddings to compute deltas; got {embeddings.shape[0]}"
        )
    # Output dtype float64 (D-31). NOTE: numpy 2.x changed np.sum semantics so that
    # axis-reductions of float32 stay float32 (unlike the float64-by-default behavior
    # of older numpy versions referenced in D-22). Cast to float64 explicitly to keep
    # the contract Phase 3 / Phase 5 expect for downstream MAD/score arithmetic.
    dots = np.sum(embeddings[:-1] * embeddings[1:], axis=1, dtype=np.float64)
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


def score_index_to_timestamp(score_idx: int, timestamps: np.ndarray) -> float:
    """The ONLY index→seconds conversion site for delta/score arrays (D-24 / Pitfall 8).

    ``score[i]`` / ``smoothed[i]`` / ``delta[i]`` all align to ``timestamps[i+1]`` —
    the delta represents change ARRIVING AT frame i+1 (spec §4 closing note).
    Phase 3 (clip_selection.py) MUST import this function rather than computing
    ``timestamps[idx + 1]`` inline.

    Returns a Python float (not numpy scalar) so JSON serialization downstream
    (Phase 5) is clean.
    """
    return float(timestamps[score_idx + 1])


def smooth_deltas(raw_deltas: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median-filter raw deltas with edge-preserving boundary handling (D-27 / Pitfall 9).

    DEVIATION from spec §5 verbatim: spec calls the ``scipy.signal`` medfilt
    function which zero-pads edges and produces phantom dips at the first/last 2
    samples (Pitfall 9 / research/PITFALLS.md §9). We use
    ``scipy.ndimage.median_filter(mode='reflect')`` instead — same kernel-5 median
    in the interior, reflected boundaries preserve edge values.

    Args:
        raw_deltas: 1-D array of cosine deltas (typically the output of compute_deltas).
        kernel_size: odd integer; default 5 per spec §5.

    Returns:
        1-D float64 array, same shape as input.
    """
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd; got {kernel_size}")
    return median_filter(raw_deltas, size=kernel_size, mode="reflect")


def mad_normalize(smoothed: np.ndarray, window_samples: int = 180) -> np.ndarray:
    """Rolling MAD normalization over a centered window (D-29 / Pitfall 10).

    Implements spec §5 step 2 with one DEVIATION: MAD floor is 1e-3 (raised from
    the spec's much-looser float-precision guard — Pitfall 10 / research/PITFALLS.md §10).
    Static-footage windows produce MAD values in the [1e-7, 1e-3] range that round-trip
    through float math and produce ceiling-pegged scores when used as a divisor; 1e-3
    is well below typical body-cam baseline noise (~0.01) but well above
    float-precision artifacts.

    Output is clipped to [0.0, 10.0] per spec §5 and dtype is float64 (D-31).

    Args:
        smoothed: 1-D array (typically the output of smooth_deltas).
        window_samples: centered window full-width in samples; default 180 (= 90s at 2 fps).

    Returns:
        1-D float64 array, same shape as input, all values in [0.0, 10.0].
    """
    n = len(smoothed)
    half = window_samples // 2  # 90 for default 180
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        local = smoothed[lo:hi]
        local_median = float(np.median(local))
        # scale=1.0 → raw MAD (no consistency factor 1.4826); matches spec §5 intent
        # of "number of MADs above local median". scipy 1.16.2 default is scale=1.0
        # but we pin it explicitly so future scipy version bumps don't silently change.
        local_mad = float(median_abs_deviation(local, scale=1.0))
        # Pitfall 10: 1e-3 floor (raised from spec's much-looser float-precision guard) —
        # static-footage MAD ∈ [1e-7, 1e-3] poisons downstream scoring under the looser guard.
        if local_mad > 1e-3:
            out[i] = (smoothed[i] - local_median) / local_mad
        else:
            out[i] = 0.0  # static-footage window: emit zero
    return np.clip(out, 0.0, 10.0)  # spec §5


def detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0) -> list[int]:
    """PELT changepoint detection (opt-in supplementary signal — D-32 / SIGP-02).

    LAZY IMPORT: ``import ruptures`` happens inside this function body. When
    ``--pelt`` is off (the default critical path), ``ruptures`` is NEVER loaded
    into ``sys.modules``. Verified by ``assert 'ruptures' not in sys.modules``
    after a non-PELT run (D-34 step 6).

    Args:
        smoothed: (M,) array (typically the output of smooth_deltas).
        penalty: PELT penalty parameter (default 3.0 per spec §5).

    Returns:
        list[int] of changepoint indices into ``smoothed``. The trailing
        ``len(signal)`` index that ``ruptures.Pelt.predict`` always emits is dropped.
    """
    import ruptures as rpt  # LAZY IMPORT — see docstring; do NOT move to module top.
    model = rpt.Pelt(model="rbf").fit(smoothed.reshape(-1, 1))
    changepoints = model.predict(pen=penalty)[:-1]  # drop trailing len(signal)
    return [int(cp) for cp in changepoints]


if __name__ == "__main__":
    # Spec §0.5 verification harness for Phase 2 (D-34). Runs as:
    #   python signal_processing.py [<video_name>] [--save-fixture | --no-save-fixture] [--pelt]
    # Mirrors extract.py's __main__ pattern verbatim per D-34 / Phase 1 D-16.
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 §0.5 verification harness for signal_processing.py "
            "(D-34: inline __main__ — no separate test scaffolding per D-34 / Phase 1 D-16)."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help=(
            "Video name stem (e.g. 'justin_timberlake'). If omitted, picks the "
            "most-recent *_embeddings.npy under output/cache/."
        ),
    )
    parser.add_argument(
        "--save-fixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write output/cache/{video}_scores.npy (and _changepoints.npy with --pelt). "
            "Default ON; pass --no-save-fixture to skip."
        ),
    )
    parser.add_argument(
        "--pelt",
        action="store_true",
        default=False,
        help=(
            "Run detect_changepoints (lazy-imports ruptures). Off by default — "
            "verifies the lazy-import contract (D-32)."
        ),
    )
    args = parser.parse_args()

    cache = Path("output/cache")

    # ── Step 0: resolve video stem (smart default) ─────────────────────────────
    if args.video is None:
        candidates = sorted(
            cache.glob("*_embeddings.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                "[FATAL] no *_embeddings.npy found under output/cache/. "
                "Run `python extract.py <video.mp4>` first.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Strip the trailing "_embeddings" from the stem to recover the video name.
        video_name = candidates[0].stem.removesuffix("_embeddings")
        print(f"[auto] using fixture: {video_name}")
    else:
        video_name = args.video

    # ── Step 1: load fixtures ──────────────────────────────────────────────────
    emb_path = cache / f"{video_name}_embeddings.npy"
    ts_path = cache / f"{video_name}_timestamps.npy"
    if not emb_path.exists() or not ts_path.exists():
        print(
            f"[FATAL] missing fixture: {emb_path} or {ts_path}. "
            "Run `python extract.py <video.mp4>` to produce it.",
            file=sys.stderr,
        )
        sys.exit(2)
    embeddings = np.load(emb_path)  # (N, 768) float32
    timestamps = np.load(ts_path)   # (N,) float64
    print(
        f"[fixture] loaded {video_name}: "
        f"embeddings={embeddings.shape} timestamps={timestamps.shape}"
    )

    # ── Step 2: compute_deltas + §0.5 print 1 (D-26) ───────────────────────────
    deltas = compute_deltas(embeddings)
    assert len(deltas) == len(embeddings) - 1, (
        f"len mismatch: {len(deltas)} vs {len(embeddings) - 1}"
    )
    assert deltas.min() >= 0.0 and deltas.max() <= 2.0, (
        f"deltas out of [0, 2]: [{deltas.min()}, {deltas.max()}]"
    )
    print(f"[deltas] first 20: {deltas[:20].tolist()}")
    print(
        f"[deltas] len={len(deltas)} "
        f"min={deltas.min():.6f} max={deltas.max():.6f} mean={deltas.mean():.6f}"
    )

    # ── Step 3: smooth_deltas + spike-injection synthetic (D-28 / Pitfall 9) ───
    smoothed = smooth_deltas(deltas)
    diff = smoothed - deltas
    print(
        f"[smoothed] (smoothed-raw) "
        f"min={diff.min():.6f} max={diff.max():.6f} mean={diff.mean():.6f}"
    )
    # Pitfall 9 spike-injection: inject +0.5 at index 100 in zeros, confirm smoothed[100]==0
    synth_z = np.zeros(200)
    synth_z[100] = 0.5
    synth_smoothed = smooth_deltas(synth_z)
    assert synth_smoothed[100] == 0.0, (
        f"[smoothed] spike at idx 100 not eliminated: got {synth_smoothed[100]} "
        "(Pitfall 9)"
    )
    # Edge-preservation (Pitfall 9): mode='reflect' preserves edges of constant signal
    edge_test = np.full(10, 7.0)
    edge_smoothed = smooth_deltas(edge_test)
    assert edge_smoothed[0] == 7.0 and edge_smoothed[-1] == 7.0, (
        f"[smoothed] edge dip — wrong boundary mode (Pitfall 9): "
        f"edges=({edge_smoothed[0]}, {edge_smoothed[-1]})"
    )
    print("[smoothed] spike-injection PASS  edge-preservation PASS")

    # ── Step 4: mad_normalize + §0.5 print 3 (D-30 / Pitfall 10) ───────────────
    scores = mad_normalize(smoothed, window_samples=180)
    # Pitfall 10 diagnostic: count windows where the MAD-floor branch actually
    # fired (mad <= 1e-3), NOT total zeros. Zeros also include negative scores
    # that the [0, 10] clip floors, which is correct behavior — only the
    # zero-MAD branch firing indicates static-footage poisoning the signal.
    half = 180 // 2
    zero_mad_branch = 0
    for i in range(len(smoothed)):
        lo = max(0, i - half)
        hi = min(len(smoothed), i + half)
        local = smoothed[lo:hi]
        if float(median_abs_deviation(local, scale=1.0)) <= 1e-3:
            zero_mad_branch += 1
    zero_mad_pct = 100.0 * zero_mad_branch / len(scores)
    above_3_pct = 100.0 * (scores > 3.0).sum() / len(scores)
    print(
        f"[scores] min={scores.min():.4f} max={scores.max():.4f} "
        f"mean={scores.mean():.4f}"
    )
    print(
        f"[scores] zero-MAD: {zero_mad_branch}/{len(scores)} "
        f"({zero_mad_pct:.2f}%); above 3.0: {above_3_pct:.2f}%"
    )
    assert scores.max() >= 2.0, (
        f"[scores] max={scores.max():.4f} < 2.0 — window too large or signal too flat (D-30)"
    )
    assert above_3_pct < 90.0, (
        f"[scores] {above_3_pct:.2f}% > 90% above 3.0 — window too small (D-30)"
    )
    assert zero_mad_pct < 5.0, (
        f"[scores] zero-MAD {zero_mad_pct:.2f}% > 5% — static-footage dominating (Pitfall 10)"
    )
    assert scores.min() >= 0.0 and scores.max() <= 10.0, (
        f"[scores] not clipped to [0, 10]: [{scores.min()}, {scores.max()}]"
    )

    # ── Step 5: synthetic alignment test (D-25 / Pitfall 8 — THE answer to spec §4) ──
    # Build (K, 768) two-color fixture: rows 0..K-1 = unit vector e_0,
    # rows K..end = unit vector e_1. Cosine delta is 0 except at score-index K-1
    # where it's 1.0. Smoothing CAN attenuate a single isolated spike under k=5
    # median, so we assert the alignment invariant against the RAW deltas argmax —
    # this is the locked behavior per Pitfall 8 / plan-checker (the helper's
    # correctness is what's being verified, not whether smoothing preserves a
    # single-sample event).
    K = 7   # transition at frame 7
    N_synth = 14
    e_a = np.zeros(768, dtype=np.float32); e_a[0] = 1.0
    e_b = np.zeros(768, dtype=np.float32); e_b[1] = 1.0
    synth_emb = np.stack([e_a] * K + [e_b] * (N_synth - K))  # (14, 768)
    synth_ts = np.arange(N_synth, dtype=np.float64) * 0.5    # 2 fps timestamps
    # Round-trip through compute_deltas → smooth_deltas (smoothing attenuation
    # of the K=7 single-sample spike is intentional and documented).
    synth_deltas = compute_deltas(synth_emb)                 # (13,)
    _synth_smoothed = smooth_deltas(synth_deltas)            # (13,) — k=5 median attenuates the spike
    # Align against the raw-deltas peak (Pitfall 8 alignment invariant):
    peak_idx = int(np.argmax(synth_deltas))
    assert peak_idx == K - 1, (
        f"[alignment] peak at {peak_idx}, expected {K - 1} (Pitfall 8)"
    )
    peak_ts = score_index_to_timestamp(peak_idx, synth_ts)
    expected_ts = K * 0.5  # frame K is the first 'arrived at' frame of color B
    assert abs(peak_ts - expected_ts) < 1e-9, (
        f"[alignment] ts={peak_ts} expected={expected_ts} "
        f"delta={abs(peak_ts - expected_ts)} (ε=1e-9)"
    )
    print(
        f"[alignment] PASS  peak at score-idx {peak_idx} → ts {peak_ts}s "
        f"(expected {expected_ts}s)"
    )

    # ── Step 6: lazy-import verification (D-32 / Pitfall enforcement) ──────────
    assert "ruptures" not in sys.modules, (
        "[lazy-import] ruptures was loaded without --pelt — D-32 violated"
    )
    print("[lazy-import] PELT clean  ('ruptures' not in sys.modules)")

    # ── Step 7: optional --pelt branch (D-32, D-33) ────────────────────────────
    changepoints = None
    if args.pelt:
        changepoints = detect_changepoints(smoothed, penalty=3.0)
        assert "ruptures" in sys.modules, (
            "[lazy-import] --pelt set but ruptures still not loaded — bug in detect_changepoints"
        )
        assert isinstance(changepoints, list) and all(
            isinstance(c, int) for c in changepoints
        ), (
            f"[pelt] return type wrong: {type(changepoints)} of "
            f"{type(changepoints[0]) if changepoints else None}"
        )
        print(f"[pelt] {len(changepoints)} changepoints; first 10: {changepoints[:10]}")

    # ── Step 8: write fixture (D-35) ───────────────────────────────────────────
    if args.save_fixture:
        from utils import ensure_output_dirs
        paths = ensure_output_dirs(video_name)
        cache_dir = paths["cache"]
        scores_path = cache_dir / f"{video_name}_scores.npy"
        np.save(scores_path, scores.astype(np.float64))
        print(f"[fixture] wrote {scores_path}")
        if args.pelt and changepoints is not None:
            cp_path = cache_dir / f"{video_name}_changepoints.npy"
            np.save(cp_path, np.asarray(changepoints, dtype=np.int64))
            print(f"[fixture] wrote {cp_path}")

    # ── Step 9: final banner ───────────────────────────────────────────────────
    print("Phase 2 §0.5 verification: PASS")
