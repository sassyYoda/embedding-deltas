# Phase 1: Frame Extraction & Embeddings - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Mode:** `--auto` (recommended option auto-selected for each gray area; choices logged inline below and in `01-DISCUSSION-LOG.md`)

<domain>
## Phase Boundary

Produce the canonical `(timestamps, embeddings)` artifact from one sample body cam video, with the 8 highest-blast-radius frame/embedding pitfalls neutralized so every downstream phase trusts its input. Specifically, this phase delivers:

- `extract.sample_frames(video_path, fps=2.0) -> (frames, timestamps)` driving sampling via `CAP_PROP_POS_MSEC`.
- `extract.load_model() -> (model, preprocess, device)` and `extract.embed_frames(frames, model, preprocess, device, batch_size=32) -> embeddings (N, 768) float32, L2-normalized`.
- `utils.probe_video_metadata(path) -> dict` (ffprobe-based duration + stream info).
- `utils.ensure_output_dirs(video_name) -> dict[str, Path]` (creates `output/{reels,clips/<name>,timestamps}` as needed).
- An `if __name__ == "__main__":` block in `extract.py` that runs the spec §0.5 print/assert verification on a single sample video.
- Project skeleton (spec §1 layout) + pinned `requirements.txt` + `ffmpeg` precondition check.

**Not in this phase:** signal processing (Phase 2), clip selection (Phase 3), ffmpeg cut/concat (Phase 4), JSON assembly + determinism env vars (Phase 5), batch run + README (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Project Skeleton & Dependencies

- **D-01:** Project layout matches spec §1 verbatim — `pipeline.py`, `extract.py`, `signal_processing.py`, `clip_selection.py`, `export.py`, `utils.py`, `videos/`, `output/{reels,clips,timestamps}/`, `requirements.txt`. (Reference ENV-03.)
- **D-02:** `requirements.txt` uses the research-pinned versions: `torch==2.11.0`, `torchvision==0.26.0`, `open_clip_torch==3.3.0`, `opencv-python==4.13.0.92`, `numpy>=2.1,<2.5`, `scipy==1.16.2`, `ruptures==1.1.10`, `tqdm>=4.67,<5`. **`ffmpeg-python` is NOT included** (unmaintained since 2019; spec uses `subprocess.run` directly). Rationale lives as a comment at top of `requirements.txt`. (Reference ENV-01, research/STACK.md.)
- **D-03:** System `ffmpeg` precondition is checked at the start of `extract.__main__` and at the start of `pipeline.run()` (in Phase 5). For Phase 1: `subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)` wrapped in a clear error message — fail fast, don't proceed silently. (Reference ENV-02.)

### Frame Sampling Strategy

- **D-04:** Frame seeking uses `cv2.VideoCapture` with **`CAP_PROP_POS_MSEC`** time-based seek, NOT `CAP_PROP_POS_FRAMES`. This neutralizes Pitfall 2 (frame-index seek lands on nearest preceding keyframe on VFR body-cam MP4s). Per-frame timestamps are read back via `cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0` to record the *actual* sampled time, not nominal `idx / fps`. (Reference EXTR-03, research/PITFALLS.md §2.)
- **D-05:** Video duration is sourced via `ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0`, NOT `cv2.CAP_PROP_FRAME_COUNT / fps` (Pitfall 3 — `CAP_PROP_FRAME_COUNT` is unreliable on body-cam encodings). `utils.probe_video_metadata` is the single owner of this call. (Reference EXTR-04.)
- **D-06:** No upfront probe-then-switch logic for the 5 sample videos. The `CAP_PROP_POS_MSEC` path is correct for both CFR and VFR; running it unconditionally costs nothing. **Rejected alternative:** running `ffprobe -show_streams` first and switching seek strategy based on `nb_frames` divisibility — adds branches for no benefit. (Auto-selected: option 1 of Area C.)
- **D-07:** BGR→RGB conversion is done inside `sample_frames` via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` (Pitfall 1). Returned frame array is RGB `uint8 (H, W, 3)`.
- **D-08:** §0.5 sanity assertion at end of sampling: `assert abs(timestamps[-1] - probe_duration) < 1.0` — guards against silent frame-count drift. Print: `"Sampled {N} frames; first 5 timestamps: {ts[:5]}; last timestamp: {ts[-1]:.3f}s; ffprobe duration: {probe_duration:.3f}s"`. (Reference EXTR-05.)

### Embedding Extraction

- **D-09 (amended 2026-05-06 after Phase 1 execution):** Model is loaded via `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', force_quick_gelu=True)`. The `force_quick_gelu=True` kwarg is **required** in `open_clip_torch==3.3.0` — the bare call (model name + `pretrained='openai'` only) builds a standard-GELU model and silently mismatches the OpenAI pretrained weights (open_clip emits a `UserWarning: QuickGELU mismatch` at load time). Spec §2 explicitly relies on the QuickGELU activation for the Koddenbrock 2025 robustness justification, so the kwarg is non-negotiable. The original RESEARCH.md / CONTEXT claim that "the registry forces QuickGELU" was wrong — verified against `factory.py` line 446–451: `force_quick_gelu` controls `model_cfg["quick_gelu"]`, the registry tag does not. (Reference EMBD-01, research/STACK.md, fix commit landing post-Plan-01-02.)
- **D-10:** Model loading is **factored out into `extract.load_model() -> (model, preprocess, device)`**, not done inside `embed_frames`. Callers (Phase 1's `__main__`, Phase 5's `pipeline.run()`, Phase 6's batch loop) load the model once and pass it in. **Rejected alternative:** spec §2 snippet loads the model at module top-level — that pattern hurts Phase 6's multi-video batch run (~30 s model load × 5 = wasteful). The signature change is invisible to spec §2's intent. (Auto-selected: option 1 of Area B.)
- **D-11:** Device selection is centralized in `load_model()`: `torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')`. The inference loop in `embed_frames` has **no device-specific branches** — `frames.to(device)` is the only place device shows up. Spec §1's "no GPU-specific code paths" intent is preserved. (Auto-selected: option 1 of Area A. Reference ENV-04, research/STACK.md Mac/Apple Silicon notes.)
- **D-12:** `embed_frames(frames, model, preprocess, device, batch_size=32)` is the **single batched code path**. No single-frame fallback (Pitfall 7). PIL conversion via `Image.fromarray(rgb_array)` before `preprocess` (Pitfall 5). Inference wrapped in `model.eval()` + `torch.inference_mode()` (Pitfall 4). Batches of exactly 32 except the trailing partial batch. (Reference EMBD-02, EMBD-03.)
- **D-13:** L2 normalization is **explicit and asserted**, not assumed: `feats = feats / feats.norm(dim=-1, keepdim=True)`, then `assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5), "embeddings not L2-normalized"` — a hard assertion, not a print. open_clip's `encode_image` does NOT normalize by default; without this guard, the spec's `np.clip(dots, -1, 1)` in Phase 2 would mask the bug by flatlining deltas. (Reference EMBD-05, research/PITFALLS.md §6.)
- **D-14:** Output embeddings: `np.ndarray` shape `(N, 768)`, dtype `float32`. Returned to caller; no internal mutation. (Reference EMBD-04.)
- **D-15:** Determinism precursors that belong in Phase 1 (the env-var stanza is Phase 5's responsibility): `model.eval()`, `torch.inference_mode()`, single batched code path, `torch.backends.cudnn.benchmark = False` set inside `load_model()`. No env vars set here. (Auto-selected: option 1 of Area H.)

### Verification Harness

- **D-16:** §0.5 verification lives in `extract.py`'s `if __name__ == "__main__":` block, runnable as `python extract.py <sample.mp4>`. It prints the sampled-frame summary (D-08), the embedding shape, and the L2-norm assertion result; runs only on a single video. **Rejected alternative:** a separate `verify.py` orchestrator — adds scaffolding for no benefit at this phase. (Auto-selected: option 1 of Area F.)
- **D-17:** Bit-identical-rerun check (Pitfall 7 closer): the `__main__` block embeds the same single frame twice via `embed_frames(frame, model, preprocess, device, batch_size=32)` and asserts the two output vectors are equal byte-for-byte. Exposes nondeterminism early (CUDNN, MPS, etc.). (Reference EMBD-06.)

### Fixture Artifact for Downstream Phases

- **D-18:** When the `__main__` verification block runs, it **also writes** `output/cache/{video_name}_embeddings.npy` and `output/cache/{video_name}_timestamps.npy` as a side-effect (gated behind `--save-fixture` flag, default ON in Phase 1's `__main__`). This unblocks Phase 2/3 development without re-running 30+ minutes of CLIP inference. The cache is technically v2 (DIAG-02 in REQUIREMENTS.md) but the *fixture-writing* portion is a Phase 1 dev-productivity concern, not the user-facing `--cache` flag. The flag is internal and not added to `pipeline.py`. (Auto-selected: option 1 of Area E. Reference research/SUMMARY.md "Build-order parallelism".)
- **D-19:** **No user-facing embedding cache.** The DIAG-02 v2 feature (`--cache` flag on `pipeline.py` for re-running with different parameters) is **deferred entirely**. The Phase 1 fixture writes are an internal dev tool. (Auto-selected: option 1 of Area G.)

### `utils.py` API Surface (Phase 1 subset)

- **D-20:** Phase 1 ships exactly these `utils.py` functions:
  - `probe_video_metadata(video_path: str | Path) -> dict` — returns `{"duration_sec": float, "width": int, "height": int, "codec": str, "is_vfr": bool}`. Uses `ffprobe -show_streams -show_format -of json`. Raises a clear error if ffprobe is missing or the file is not a video.
  - `ensure_output_dirs(video_name: str, base: Path = Path("output")) -> dict[str, Path]` — creates `reels/`, `clips/{video_name}/`, `timestamps/`, and `cache/` (for D-18); returns a dict of paths.
  - `setup_logger(name: str = "highlight_reel") -> logging.Logger` — minimal stdlib logger with `[%(levelname)s] %(message)s` format. Used by `extract.py` for the §0.5 prints; `print()` is also acceptable per spec — pick one and use it consistently.
- **D-21:** `utils.write_timestamps_json` is **deferred to Phase 5** where the JSON schema is assembled and the strict three-state PELT field is enforced. (Auto-selected: option 1 of Area D.)

### Claude's Discretion

- Logger vs `print()` for §0.5 verification prints — D-20 offers a logger but spec snippets use `print()`. Either is acceptable as long as it's consistent within `extract.py`. Default to `print()` to match spec §0.5 verbatim; switch to logger only if Phase 5 introduces structured logging.
- Exact wording of error messages, docstrings, and type-hint style — match Python ≥3.11 conventions (PEP 604 unions like `str | Path`, lowercase `dict[str, Path]`, etc.).
- Tolerance constants (`atol=1e-5` for L2 norm, `< 1.0` for duration drift) — these are sensible defaults grounded in research; planner / executor can tighten if a sample video reveals issues.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, executor) MUST read these before planning or implementing.**

### Locked Spec
- `assignment-details/bodycam_highlight_reel_spec.md` §0 (Orientation), §0.5 (Testing Protocol), §1 (Environment & Dependencies), §2 (Embedding Model), §3 (Frame Sampling), §10 (Key Design Decisions Summary), §12 (What Not To Do) — single source of truth; treat as authoritative.

### Project-Level
- `.planning/PROJECT.md` — Core Value, Active requirements, Constraints, Key Decisions table.
- `.planning/REQUIREMENTS.md` — REQ-IDs ENV-01..04, EXTR-01..05, EMBD-01..06 (15 mapped to Phase 1).
- `.planning/ROADMAP.md` Phase 1 detail block — goal, dependencies, success criteria, plan placeholder.
- `.planning/STATE.md` — current memory.

### Research
- `.planning/research/SUMMARY.md` — Executive summary; "Implications for Roadmap" section maps decisions to phases. **Read first.**
- `.planning/research/STACK.md` — Pinned versions, `open_clip` registry confirmation, scipy 1.15 medfilt regression flag, Mac/Apple Silicon device notes, full `requirements.txt` rationale.
- `.planning/research/ARCHITECTURE.md` — Module-level data contracts, `embed_frames(model, preprocess)` rationale, build-order DAG, `utils.py` proposed API.
- `.planning/research/PITFALLS.md` — 20 pitfalls; **Phase 1 owns Pitfalls 1, 2, 3, 4, 5, 6, 7, 15**. Each pitfall has failure mode, warning sign, and one-line guard. Read these before writing extract.py.

### External (referenced by research; URL-only — no local copies)
- `https://github.com/mlfoundations/open_clip` — `create_model_and_transforms` API and pretrained registry.
- `https://github.com/scipy/scipy/issues/22333` — medfilt regression in scipy 1.15 (motivates `==1.16.2` pin).
- PyTorch reproducibility docs — `torch.use_deterministic_algorithms`, `cudnn.benchmark` semantics.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

None. This is a greenfield Python project. No `src/`, `app/`, or library modules exist yet. The only repo content prior to Phase 1 is `assignment-details/` (the locked spec + PDF) and `videos/` (the gitignored sample zip).

### Established Patterns

None — Phase 1 *establishes* the patterns the rest of the project will follow:

- **Module orchestration pattern:** each module exposes pure functions; `pipeline.py` (Phase 5) is the only orchestrator with side effects on disk + subprocess.
- **Index-space vs time-space separation:** locked starting in Phase 2, but Phase 1's data contracts (`(frames, timestamps)` from `sample_frames`; `embeddings` from `embed_frames`) make this possible. **`signal_processing.py` will receive `embeddings` only — never `timestamps`** (Phase 2 enforces this).
- **§0.5 verification gate:** each module ships an `if __name__ == "__main__":` block that runs the spec-mandated checks; passing those is the gate to start the next phase.
- **Lazy-import pattern for opt-in dependencies:** Phase 2 will lazy-import `ruptures`. Phase 1 has no equivalent — all its deps are required.

### Integration Points

- **Phase 1 → Phase 2:** `embeddings: np.ndarray (N, 768) float32 L2-normalized` flows into `signal_processing.compute_deltas(embeddings)`. The fixture cache (D-18) is the dev-time integration point.
- **Phase 1 → Phase 3:** `timestamps: np.ndarray (N,) float64 seconds` flows into `clip_selection.build_clips(...)` via Phase 5's orchestrator. **Phase 3 receives `timestamps` directly; Phase 2 does not.**
- **Phase 1 → Phase 5:** `extract.load_model()`, `extract.sample_frames()`, `extract.embed_frames()`, `utils.probe_video_metadata()`, `utils.ensure_output_dirs()` are all called from `pipeline.run()`.
- **Phase 1 → Phase 6:** `load_model()` is called **once** by the batch driver; `(model, preprocess, device)` is passed into each video's `embed_frames` invocation. This amortizes the ~30s model load (D-10 rationale).

</code_context>

<specifics>
## Specific Ideas

- **Spec §0.5 prints are the success criteria, not the comments.** Treat the print/assert lines literally — don't paraphrase. Reviewers will look at `extract.py`'s `__main__` and expect to see them.
- **The bit-identical-rerun check (D-17)** is a small addition beyond spec §0.5 — it catches the determinism class of bugs early (Pitfall 15) so Phase 5's "two-run identical JSON" acceptance test isn't the first place they surface.
- **The `--save-fixture` flag (D-18)** is dev-only. It MUST NOT appear in `pipeline.py`'s argparse. It is invoked only via `python extract.py <video> --save-fixture`.
- The research's `embed_frames(model, preprocess)` signature deviation from spec §2's snippet is **intentional and locked** (D-10). Plans and reviews should not "fix" it back to module-level loading.

</specifics>

<deferred>
## Deferred Ideas

These came up during analysis but belong in other phases or are explicitly out of scope:

- **Determinism env-var stanza** (`OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`) → Phase 5 (`pipeline.py` entry). Reference: ROBU-02 (v2) in REQUIREMENTS.md.
- **User-facing `--cache` flag** for `pipeline.py` → v2 (DIAG-02). Phase 1's fixture write is internal dev only.
- **`utils.write_timestamps_json`** → Phase 5, where the JSON schema is assembled and the PELT three-state field is enforced.
- **`--diagnostics` plots** (PNG of raw deltas, smoothed, MAD, peaks) → v2 (DIAG-01). Out of scope for v1.
- **Concat-filter re-encode fallback** → Phase 4 (`export.py`). Reference: ROBU-01 (v2) but the fallback path itself is in v1 EXPC-02 implicitly.
- **JSON float rounding (3 dp seconds, 4 dp scores)** → Phase 5. Reference: ROBU-03 (v2).
- **Probe-then-switch seek strategy based on VFR detection** — rejected (D-06). The single `CAP_PROP_POS_MSEC` path is correct for both.
- **Module-level model loading per spec §2 verbatim** — rejected (D-10). Caller-passes-model is the locked signature.

</deferred>

---

*Phase: 01-frame-extraction-embeddings*
*Context gathered: 2026-05-06*
*Mode: --auto (single pass; recommended option auto-selected per gray area)*
