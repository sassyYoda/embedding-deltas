# Feature Research

**Domain:** Visual-embedding-only video highlight extraction (CLI prototype, take-home assignment)
**Researched:** 2026-05-06
**Confidence:** HIGH (every feature traces directly to the locked spec at `assignment-details/bodycam_highlight_reel_spec.md` or to standard reviewer expectations for a coding take-home)

---

## Orientation

This is **not** a product market. This is a **single-reviewer evaluation** of a take-home prototype for AbelPolice. The "users" are:

1. **The reviewer** — wants to see defensible design choices, working code, honest analysis
2. **The pipeline operator** — runs `python pipeline.py video.mp4` and expects coherent output

Therefore "table stakes" = "what makes the submission count as complete per the spec rubric"; "differentiators" = "what would make a reviewer rank this submission above an equally-correct one"; "anti-features" = "spec violations + scope creep that signals poor judgment."

Every feature below is tagged with its provenance (spec section or take-home convention) and complexity (S = hours, M = ~half day, L = full day or more, given the spec already locks the algorithm).

---

## Feature Landscape

### Table Stakes (Spec-Mandated — Required for "Complete")

These come straight from the spec. Missing any of them = the submission fails the rubric or violates the assignment.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **2 fps frame sampling via OpenCV seek** | Spec §3 — non-negotiable rate; seeking (not full decode) is mandated | S | `cap.set(CAP_PROP_POS_FRAMES, …)` per sample; do not decode every frame |
| **CLIP ViT-L/14 OpenAI/QuickGELU embedding extraction** | Spec §2 — model is locked; no substitution allowed | S | `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')`, `model.eval()`, batch=32, `torch.no_grad()` |
| **L2-normalized 768-d embeddings, batched** | Spec §2 — embedding output contract | S | Verifiable via `np.allclose(np.linalg.norm(E, axis=1), 1.0)` |
| **Cosine-distance delta computation** | Spec §4 — delta metric is locked | S | Dot product of normalized vectors, clipped to [-1,1], then `1 - dot` |
| **Median filter (kernel=5) on raw deltas** | Spec §5 step 1 — order is locked: filter BEFORE normalize | S | `scipy.signal.medfilt(deltas, kernel_size=5)` |
| **Rolling MAD normalization (180-sample / 90s window)** | Spec §5 step 2 — replaces global threshold; core of the approach | M | Centered window, guard against zero-MAD windows, post-clip to [0,10] |
| **Peak detection via `scipy.signal.find_peaks`** | Spec §6 — height + distance gating | S | `--height` default 1.5, `--min-gap-sec` default 15 → `min_gap_samples = min_gap_sec * fps` |
| **Adaptive padding `max(3, min(8, budget × 0.15))`** | Spec §6 — prevents short videos from blowing budget | S | Pure arithmetic |
| **Duration budget `(video_dur / 1800) × 60`** | Spec §6 — 1 min reel per 30 min source | S | One-liner |
| **Build → merge (gap=3s) → enforce-budget pipeline** | Spec §6 — order is mandatory; merge before budget check | M | Greedy by score during budget enforcement; chronological re-sort for export |
| **`ffmpeg -c copy` per-clip extraction + concat demuxer** | Spec §7 — re-encoding is explicitly discouraged | S | `subprocess.run` with `-ss/-to/-i/-c copy`; concat manifest file |
| **JSON timestamp output matching exact §8 schema** | Spec §8 — schema is locked field-for-field | S | Must include `embedding_model`, `sampling_fps`, `mad_score`, `raw_cosine_delta`, `coincides_with_pelt_changepoint` (null when `--pelt` off) |
| **Single CLI entry point `pipeline.py <video>`** | Spec §9 — one command per video | S | argparse, positional `video`, flags `--pelt`, `--height`, `--min-gap-sec` |
| **Project structure (extract/signal_processing/clip_selection/export/utils + pipeline)** | Spec §1 — module layout is locked | S | Reviewer will look for these files; don't collapse into a monolith |
| **Output directory layout `output/{reels,clips,timestamps}/`** | Spec §1 + §8 | S | Auto-create with `os.makedirs(..., exist_ok=True)` |
| **Optional `--pelt` flag with ±5-sample / 1.2× boost** | Spec §5 + §6 — opt-in supplementary signal | M | `ruptures.Pelt(model='rbf')`, only imported when flag is set; null in JSON otherwise |
| **Runs on all 5 sample videos with one fixed parameter set** | Spec §6 + §0 — algorithmic-only requirement | S (procedural; M if tuning is hard) | Pick parameters on the most representative video, freeze, run the rest |
| **`requirements.txt`** | Spec §1 implicit; reviewer needs to reproduce | S | Pin majors at minimum: `torch`, `open_clip_torch`, `opencv-python`, `numpy`, `scipy`, `ruptures`, `ffmpeg-python`, `tqdm` |
| **README with run instructions** | Take-home convention; reviewer needs to know how to invoke | S | One section: install, ffmpeg dep, command; that's the floor |
| **Per-module test prints (spec §0.5 protocol)** | Spec §0.5 — explicit testing protocol | S | First 5 timestamps, embedding shape, L2-norm check, raw delta range, MAD min/max/mean, peak count + timestamps, clip durations, total-vs-budget assertion |

**Subtotal:** ~20 spec-mandated features. None can be cut. Most are S complexity because the spec already specifies the implementation.

---

### Differentiators (Above-Bar Submission Quality)

These are not in the spec rubric but separate a "complete" submission from a "thoughtful" one. The spec itself signals this in §0 ("flag explicitly before proceeding") and §11 ("honest analysis of limitations beats optimistic framing"). The reviewer is Antonio + AbelPolice; both have explicitly invited candidate judgement, so demonstrating it is rewarded.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Per-module verification script that emits the §0.5 prints automatically** | Reviewer can re-run testing protocol without piecing it together; demonstrates discipline | S | A single `verify.py` (or `--verify` flag on each module) that hits all §0.5 checks. Strictly stronger than ad-hoc prints. |
| **Reproducibility script `run_all.sh` (or `python -m run_all`)** | Reviewer runs one command, gets all 5 reels + 5 JSONs; no manual loop | S | The spec's `for f in videos/*.mp4; do …` line, formalized. Print per-video runtime. |
| **README "Design Rationale" section linking each parameter to the spec** | Shows the candidate read and internalized the spec, didn't just transcribe code | S | One paragraph each for: model choice (Koddenbrock), 2 fps, median-before-MAD, 90s window, adaptive padding, `-c copy` |
| **README "Known Limitations" section mirroring spec §11** | Honest framing the spec explicitly endorses ("honest analysis beats optimistic framing") | S | Visually-distinct ≠ important; lighting transitions; classification→delta robustness gap; no labeled GT |
| **README "Per-Video Notes" qualitative writeup** | Replaces missing precision/recall — only credible eval given no ground truth | M | One paragraph per video: what kinds of moments got picked, false positives observed, plausible explanation rooted in the algorithm |
| **Sanity-check plots saved as PNGs to `output/diagnostics/{video}/`** | Reviewer can scan a single image to see the signal looks healthy | M | (1) raw delta + smoothed delta overlay, (2) MAD-normalized score with peak markers + threshold line, (3) optional PELT changepoints overlaid. Use matplotlib; do NOT add matplotlib if not desired — make it opt-in via `--diagnostics` flag |
| **Deterministic seeding + `torch.set_num_threads(N)` log line** | Reproducibility across runs — reviewer can re-run and get the same JSON | S | CLIP inference is deterministic given fixed input order, but log device + thread count anyway |
| **Progress bars via `tqdm` on frame sampling + embedding** | Reviewer running locally on CPU sees that something is happening across thousands of frames | S | Already in spec deps; just wire it in |
| **Structured logging with phase headers `[1/6] ... [6/6]`** | Spec §9 already shows this format; just respect it | S | Cheap and aligns with what spec author wrote |
| **Graceful handling of corrupt frames / empty reads** | Body cam footage has glitches; not crashing on one bad frame is robustness | S | Skip frame if `cap.read()` returns `False`; advance `frame_idx`; log count of skipped frames |
| **Idempotent re-runs (skip if `output/reels/{name}_highlight.mp4` exists unless `--force`)** | Reviewer iterates without re-doing the slow embedding step | S | `--force` flag + existence check |
| **Optional embedding cache `output/cache/{video}_embeddings.npy`** | Re-running signal-processing experiments without re-doing 30 min of CLIP inference | M | Hash by file path + size + mtime; spec doesn't forbid this and it dramatically speeds tuning |
| **`--dry-run` flag that runs everything except ffmpeg export** | Lets the reviewer iterate on parameters without writing video files | S | Prints final clip list + JSON only |
| **Total runtime + per-phase timing in stdout** | Reviewer can see where time goes; signals the candidate cares about performance | S | `time.perf_counter()` deltas around each `[N/6]` block |
| **Type hints on all module-level functions** | Modern Python convention; spec shows hints in code samples | S | Already partially modeled in spec snippets |
| **Single-paragraph "Why not X?" subsection in README** | Reviewer wonders why no DBSCAN clustering, no scene detection by color histogram, no transformer-based summarization. Briefly acknowledging and dismissing each shows breadth | S | 3-4 alternatives, 1 sentence each |
| **CLI `--help` text with parameter rationale** | argparse `help=` strings that say *why* a default is what it is, not just what the flag does | S | Free differentiator — costs ~30 seconds, signals care |

**Subtotal:** ~17 differentiators. Pick ~6–10. The diagnostic plots, the design-rationale README, the run-all script, the per-video notes, and the embedding cache are the highest-impact-per-hour. Skip the rest if time is tight.

---

### Anti-Features (Explicitly Avoid)

These are either spec violations (cite §12) or scope creep that would signal poor judgement on a take-home.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Audio / ASR / transcript signals in selection** | Audio is genuinely the most informative signal for body cam highlights | **Spec §12 bullet 1: "Do not use transcript, ASR, audio levels, OCR, or LLMs for clip selection — this violates the assignment constraints"** | Visual embeddings only; document the loss in README §Limitations |
| **OCR on frames (timecode, weapon serials, etc.)** | Body cams overlay timestamps — tempting signal | **Spec §12 bullet 1** explicitly forbids OCR | None — visual embedding deltas only |
| **LLM-based scene captioning or clip selection** | Tempting modern approach; "describe this frame" → score | **Spec §12 bullet 1** explicitly forbids LLMs in selection path | None — visual embedding deltas only |
| **Manual timestamp curation, even for sanity-checking** | Quick way to validate the pipeline emits something sensible | **Spec §12 bullet 2: "Do not manually select timestamps even to verify the pipeline — clip selection must be fully algorithmic"** | Watch the auto-generated reels; document qualitatively |
| **Per-video parameter retuning** | Each video has different action density; tempting to tune `--height` per file | **Spec §6: "Do not retune per-video — that would amount to manual selection, which violates the algorithmic requirement"** | Tune once on the most representative video, freeze, run all 5 |
| **Fine-tuning or modifying CLIP weights** | Domain adaptation could plausibly help | **Spec §2: "Do NOT fine-tune or modify the model in any way"** | Use OpenAI pretrained weights as-is |
| **Substituting a different embedding model (SigLIP, DINOv2, etc.)** | Newer = better, naively | **Spec §2: "This is non-negotiable"** — choice is grounded in Koddenbrock 2025 robustness data | CLIP ViT-L/14 OpenAI/QuickGELU only |
| **Global fixed-threshold peak detection** | Simpler than rolling MAD | **Spec §12 bullet 3: "Do not use a fixed global threshold instead of rolling MAD — this was the weakness of Antonio's original k-most-distinct approach"** | Rolling MAD with 90s window |
| **Median-filter applied to raw video frames** | Plausible misreading of "median filter" instruction | **Spec §12 bullet 4: "Do not apply the median filter to the raw video frames — filter the 1D delta signal only"** | Filter the cosine-delta 1D signal only |
| **Re-encoding clips with libx264 by default** | Cleaner cuts, frame-accurate | **Spec §12 bullet 5: "Do not re-encode clips unless stream copy produces visibly broken output — re-encoding is slow and unnecessary"** | `-c copy` default; document the ~1s keyframe imprecision |
| **Web UI / Streamlit dashboard / Flask viewer** | Looks polished | Out of scope for a CLI prototype; signals scope creep + time misallocation. Spec §1 specifies a CLI structure with no UI module. PROJECT.md "Out of Scope": "Polished UI / web frontend — CLI prototype only" | CLI + JSON output + (optional) saved diagnostic PNGs |
| **Quantitative precision/recall numbers** | Reviewers love metrics | No labeled ground truth exists for these 5 videos. Inventing one would be dishonest. **Spec §11 explicitly notes this.** | Qualitative per-video writeup in README |
| **GPU-specific code paths / `if torch.cuda.is_available()` branches** | Speed | PROJECT.md "Out of Scope" + Spec §1: "Do not add GPU-specific code paths — let open_clip handle device selection" | Let `open_clip` auto-select device |
| **Configurable embedding model via CLI flag** | "Future-proofing" | Premature flexibility; the model is part of the design rationale, not a parameter. Locks reviewer attention on the wrong axis | Hardcode ViT-L/14 OpenAI; document the choice |
| **Multiprocessing / threading for frame sampling** | Faster | OpenCV + open_clip are not thread-safe across the API surface this prototype uses; introduces correctness risk for marginal speedup. Take-home is not a perf benchmark | Single-threaded; batch=32 on the GPU/CPU side already amortizes |
| **Resumable mid-pipeline checkpointing beyond embedding cache** | "Robustness" | YAGNI for ~30 min videos. Embedding cache (a differentiator above) covers the only expensive phase | Embedding cache only; restart from scratch otherwise |
| **Configurable JSON schema** | Flexibility | **Spec §8 schema is exact and locked.** Deviating risks failing the rubric | Match §8 schema field-for-field |
| **Polishing beyond what the spec asks** | "Show care" | **Spec §12 bullet 6: "Do not over-polish — a working prototype with honest analysis beats a polished system with optimistic framing"** | Working pipeline + honest README + per-module verification. Stop there. |

**Subtotal:** 18 anti-features, 11 of which trace to specific §12 bullets or other locked spec sections. Treat this list as a guardrail during implementation.

---

## Feature Dependencies

```
[CLIP ViT-L/14 model loading]
    └──requires──> [open_clip + torch installed]
    └──provides──> [embed_frames()]
                       └──requires──> [2 fps frame sampling]
                                         └──requires──> [OpenCV VideoCapture + ffmpeg system binary]

[embed_frames()]
    └──provides──> [L2-normalized 768-d embeddings]
                       └──provides──> [cosine-delta computation]
                                         └──provides──> [median filter]
                                                           └──provides──> [rolling MAD normalization]
                                                                             └──provides──> [find_peaks]
                                                                                               └──provides──> [build → merge → budget]
                                                                                                                 └──provides──> [ffmpeg extract_clip + concat]
                                                                                                                                   └──provides──> [final reel + JSON]

[--pelt flag] ──enhances──> [find_peaks output via 1.2× boost ±5 samples]
[--pelt flag] ──requires──> [ruptures installed]
[--pelt flag] ──affects──> [JSON.coincides_with_pelt_changepoint field (else null)]

[Per-module §0.5 prints] ──validate──> [each pipeline stage]
[Sanity-check plots (--diagnostics)] ──enhance──> [§0.5 prints]
[Embedding cache] ──enhances──> [iteration speed during parameter tuning]
                              ──does not affect──> [final output correctness]

[run_all.sh] ──requires──> [pipeline.py end-to-end correctness on a single video first]
                          (per spec footer: "Get one video working end to end before touching the others")

[README design rationale] ──depends on──> [spec §10 decisions table]
[README per-video notes] ──requires──> [run_all.sh having executed successfully]
[README limitations] ──mirrors──> [spec §11]
```

### Dependency Notes

- **Pipeline core is strictly linear.** extract → signal_processing → clip_selection → export → pipeline. Spec footer locks this build order. Don't parallelize implementation — earlier modules must be correct before later ones can be debugged.
- **`--pelt` is fully optional and orthogonal.** When the flag is off, `ruptures` does not need to be imported (lazy import). The JSON field becomes `null`. This isolation is important: a bug in PELT must not break the baseline pipeline.
- **§0.5 prints are not optional.** The spec elevates them from "testing" to "verification protocol" — implementing the pipeline without them is a rubric miss. Verification differentiator (`verify.py`) builds *on top of* having the prints.
- **Embedding cache is independent of correctness.** It is purely an iteration-speed feature. If it produces stale results, the bug is contained to the cache layer; correctness reviewers can disable it via `--no-cache` or by deleting `output/cache/`.
- **Per-video README notes require run-all completion.** Writeup is the last thing to do; don't draft it before seeing the actual reels.
- **Reproducibility script depends on single-video correctness.** Spec footer is unambiguous: "Get one video working end to end before touching the others." Do not write `run_all.sh` until video 1 produces a watchable reel.

---

## MVP Definition

### Launch With (Submission v1) — Strictly Spec-Mandated

These map 1:1 to the "Active" requirements in PROJECT.md. Without all of them, the submission is incomplete.

- [ ] 2 fps frame sampling (extract.py) — spec §3
- [ ] CLIP ViT-L/14 OpenAI embedding extraction, batch=32, no_grad, L2-normalized — spec §2
- [ ] Cosine-distance delta computation — spec §4
- [ ] Median filter (kernel=5) — spec §5 step 1
- [ ] Rolling MAD normalization (90s / 180-sample window), clipped [0,10] — spec §5 step 2
- [ ] `find_peaks` with `--height` (1.5) + `--min-gap-sec` (15) — spec §6
- [ ] Optional `--pelt` flag with 1.2× boost ±5 samples — spec §5 + §6
- [ ] Adaptive padding `max(3, min(8, budget × 0.15))` — spec §6
- [ ] Duration budget `(dur / 1800) × 60` — spec §6
- [ ] Build → merge (gap=3s) → enforce-budget — spec §6
- [ ] `ffmpeg -c copy` per-clip + concat demuxer — spec §7
- [ ] JSON output matching §8 schema exactly — spec §8
- [ ] CLI: `python pipeline.py <video> [--pelt] [--height H] [--min-gap-sec G]` — spec §9
- [ ] Locked module layout (extract / signal_processing / clip_selection / export / utils / pipeline) — spec §1
- [ ] Output dirs `output/{reels,clips,timestamps}/` auto-created — spec §1 + §8
- [ ] `requirements.txt` matching spec §1 deps
- [ ] README with run instructions
- [ ] Per-module §0.5 verification prints
- [ ] All 5 sample videos run successfully with one fixed parameter set

### Add After v1 Works End-to-End on One Video (Differentiators to Pick From)

Triggered by: video 1 produces a watchable reel and a valid JSON. Then layer on whichever of these fits remaining time.

- [ ] `run_all.sh` reproducibility script — trigger: video 1 works
- [ ] README "Design Rationale" section — trigger: video 1 works (write while waiting for videos 2–5 to process)
- [ ] README "Known Limitations" section — trigger: video 1 works
- [ ] README "Per-Video Notes" — trigger: all 5 videos processed
- [ ] Diagnostic PNG plots (`--diagnostics` flag) — trigger: parameter tuning is finicky and you want a visual to defend choices
- [ ] Embedding cache (`output/cache/`) — trigger: tuning requires multiple re-runs of the same video
- [ ] Idempotent skip (`--force` to override) — trigger: any of the above
- [ ] `--dry-run` flag — trigger: tuning requires many runs without writing video
- [ ] Per-phase timing logs — trigger: ~15 min of work, low risk
- [ ] argparse `help=` strings with rationale — trigger: ~15 min of work, low risk
- [ ] Type hints + docstrings — trigger: cleanup pass before submission
- [ ] "Why not X?" README subsection — trigger: cleanup pass before submission

### Future Consideration (NOT in this submission)

These exist only to document discipline — the spec/PROJECT.md explicitly defers or rejects all of them.

- [ ] Audio / ASR / OCR / LLM signals — **rejected by spec §12**, would require redesign of the assignment
- [ ] Adaptive MAD window sizing — **deferred per spec §11 limitation 5**; mention in README, do not implement
- [ ] Quantitative precision/recall — **rejected per spec §11 limitation 7**; would require labeled ground truth
- [ ] Web UI — **rejected per PROJECT.md Out of Scope**
- [ ] Per-video tuning automation — **rejected per spec §6**; would violate algorithmic-only requirement
- [ ] Alternative embedding models as configurable — **rejected per spec §2**

---

## Feature Prioritization Matrix

| Feature | Reviewer Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| Spec-mandated pipeline (all 20 table-stakes items) | HIGH (required) | MEDIUM (cumulative) | **P1 — must have** |
| Per-module §0.5 verification prints | HIGH (rubric item) | LOW | **P1 — must have** |
| README run instructions | HIGH (reviewer can't run otherwise) | LOW | **P1 — must have** |
| README design rationale section | HIGH (separates "transcribed spec" from "understood spec") | LOW | **P2 — strongly recommended** |
| README known limitations section | HIGH (spec explicitly endorses honest framing) | LOW | **P2 — strongly recommended** |
| `run_all.sh` reproducibility script | HIGH (1 command vs 5) | LOW | **P2 — strongly recommended** |
| Per-video qualitative notes in README | HIGH (only credible eval given no GT) | MEDIUM | **P2 — strongly recommended** |
| Diagnostic plots (--diagnostics) | MEDIUM (helps reviewer + helps you tune) | MEDIUM | **P2 — recommended** |
| Embedding cache | MEDIUM (speeds your iteration; reviewer doesn't see it) | MEDIUM | **P3 — if tuning is slow** |
| Idempotent re-runs / `--force` | LOW (reviewer runs once) | LOW | **P3 — nice to have** |
| `--dry-run` flag | LOW (you'll use it; reviewer won't) | LOW | **P3 — nice to have** |
| Per-phase timing logs | LOW | LOW | **P3 — nice to have** |
| Argparse rationale strings | LOW | LOW | **P3 — free, do it last** |
| Type hints + docstrings | LOW (modern convention) | LOW | **P3 — free, do it last** |
| "Why not X?" README subsection | LOW–MEDIUM (signals breadth) | LOW | **P3 — if time** |
| Anti-features (audio, OCR, LLM, web UI, etc.) | NEGATIVE | varies | **P0 — actively avoid** |

**Priority key:**
- **P0:** Actively avoid (anti-feature)
- **P1:** Must have for submission to count as complete
- **P2:** Strongly recommended; differentiates a "complete" from a "thoughtful" submission
- **P3:** Nice to have; do these only after P1+P2 are solid

---

## Comparative Reference (Take-Home Patterns, Not Product Competitors)

This is a take-home, not a market. Comparisons are against typical submission patterns, not products.

| Feature | Typical "Just Works" Submission | Typical "Above Bar" Submission | This Submission's Approach |
|---------|----------------------------------|--------------------------------|----------------------------|
| Spec adherence | Implements ~80%, skips testing protocol | Implements 100% of spec | 100% spec including §0.5 verification |
| Documentation | One-line README ("run pipeline.py") | README with rationale + limitations | README mirroring spec §10 (decisions) + §11 (limitations) + per-video notes |
| Reproducibility | "for f in videos/*; do …" in README | A run-all script | `run_all.sh` + (optional) embedding cache for fast re-runs |
| Diagnostics | None | Stdout prints | Stdout §0.5 prints + optional `--diagnostics` PNG plots |
| Honest analysis | Optimistic framing ("works great!") | Acknowledges limitations | Mirrors spec §11 + per-video qualitative notes including failure modes |
| Scope discipline | Sometimes adds a Streamlit UI to "stand out" | Sticks to CLI | Sticks to CLI; documents *why* in "Why not X?" subsection |

---

## Sources

- `assignment-details/bodycam_highlight_reel_spec.md` — locked design spec; sections §0, §0.5, §1, §2, §3, §4, §5, §6, §7, §8, §9, §10, §11, §12 (HIGH confidence — primary source of truth)
- `.planning/PROJECT.md` — derived requirements + Out of Scope list (HIGH confidence)
- Take-home submission conventions for ML/CV roles — README structure, reproducibility expectations, honest-analysis framing (MEDIUM confidence — convention, not a written rubric; but spec §11 + §12 bullet 6 explicitly endorse this framing)

---

*Feature research for: visual-embedding-only video highlight extraction CLI prototype (AbelPolice take-home)*
*Researched: 2026-05-06*
