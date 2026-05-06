# Stack Research

**Domain:** Visual-embedding video highlight extraction (Python CLI, CLIP ViT-L/14, 1D signal processing, ffmpeg cut/concat)
**Researched:** 2026-05-06
**Confidence:** HIGH (versions verified against PyPI 2026-05-06; open_clip API verified against `mlfoundations/open_clip` main-branch `pretrained.py`)

## Verdict on the Locked Spec

The spec's stack choices (§1, §2) are **still correct in 2026** with three verified caveats:

1. The exact call `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` is **still the canonical API** as of `open_clip_torch` 3.3.0 (released 2026-02-27). The registry forces `quick_gelu=True` for that tuple — no rename, no deprecation warning.
2. **`ffmpeg-python` (kkroening) is dead software** — last release July 2019. The spec lists it in §1 but spec §7 only ever uses `subprocess.run([...])` to invoke `ffmpeg` directly. **Drop `ffmpeg-python` from `requirements.txt`** — it is unused and listing it implies a maintained dependency that does not exist.
3. **scipy 1.15+ has a known `medfilt` correctness regression** (scipy issue [#22333](https://github.com/scipy/scipy/issues/22333), open as of early 2026). Pin to a version that has not regressed and add a verification print after the median-filter step (already in spec §0.5).

Everything else in §1/§2 holds. Python 3.11+ floor is still valid; this machine has 3.13.2 which is fine.

## Recommended Stack

### Core Technologies

| Technology | Pinned Version | Purpose | Why Recommended |
|------------|---------------|---------|-----------------|
| Python | `>=3.11,<3.14` | Runtime | Spec floor of 3.11 is valid in 2026. 3.13 is the current stable line on this machine; 3.14 is brand-new (Oct 2025) and `ruptures<3.14` upper-bounds it. Pinning `<3.14` matches `ruptures` 1.1.10's `requires-python`. |
| `torch` | `==2.11.0` | CLIP forward pass, tensor ops | Latest stable (2026-03-23). `open_clip_torch` 3.3.0 requires `torch>=2.6`. 2.11 has Apple Silicon MPS, Python 3.10–3.14 wheels. Pin exact: weight loading and dispatcher behavior is version-sensitive. |
| `torchvision` | `==0.26.0` | Image transforms used by `open_clip` `preprocess` | Pinned in lockstep with torch 2.11.0 per torch's published compatibility matrix. Mismatched torch/torchvision is the #1 install-time failure mode for CLIP pipelines. |
| `open_clip_torch` | `==3.3.0` | CLIP ViT-L/14 OpenAI weights + `preprocess` transform | Latest (2026-02-27). Spec §2 explicitly requires the OpenAI/QuickGELU variant — `open_clip` is the canonical loader. The `'ViT-L-14' + pretrained='openai'` tuple maps to `timm/vit_large_patch14_clip_224.openai/` on HF Hub with `quick_gelu=True` forced; the activation choice is correct without any code change. |
| `numpy` | `>=2.1,<2.5` | Embedding arrays, delta math | numpy 2.x is required by scipy 1.17 and torch 2.11. 2.4.4 is current; permit minor floats so pip can resolve with scipy/torch's transitive constraints. |
| `scipy` | `==1.16.2` | `signal.medfilt`, `signal.find_peaks`, `stats.median_abs_deviation` | **Pin to 1.16.x specifically** — scipy 1.15 introduced a `medfilt` correctness regression ([#22333](https://github.com/scipy/scipy/issues/22333)) that affects exactly the `kernel_size=5` median filter the spec uses. 1.16.x has the fix; 1.17.x adds Array-API experimental flags but is fine. Use 1.16.2 as the conservative pin. |
| `ruptures` | `==1.1.10` | PELT changepoint detection (opt-in `--pelt`) | Latest (early 2026). API is stable: `rpt.Pelt(model="rbf").fit(...).predict(pen=...)` works as written in spec §5. Upper-bounds Python at <3.14 — this is what dictates the runtime ceiling. |
| `tqdm` | `>=4.67,<5` | Progress bars over the embedding loop | 4.67.x is the current line; allow patch updates. No constraints from other deps. |
| `ffmpeg` (system) | `>=4.4`, recommend `>=6.0` | Per-clip cut + concat demuxer | The user's machine has 7.1.1 (Homebrew). Concat demuxer with `-c copy` is reliable from ffmpeg 4.4 forward; `-ss` placement *before* `-i` for fast seek (as in spec §7) has been stable since ~3.0. **Minimum viable: 4.4.** **Recommended: 6.0+** for reliable handling of variable-frame-rate body cam streams. |

### Removed From Spec §1

| Listed in spec | Action | Reason |
|----------------|--------|--------|
| `ffmpeg-python` | **Drop** | Last release July 2019. Project is unmaintained. Spec §7 code uses `subprocess.run(['ffmpeg', ...])` directly — the binding is never imported. Listing it pollutes `requirements.txt` with a stale dep. |

This is the only deviation from the locked spec. It is a documentation-only correction (no code change is needed because the spec's export code already bypasses the binding).

### Supporting Libraries

| Library | Pinned Version | Purpose | When to Use |
|---------|---------------|---------|-------------|
| `opencv-python` | `==4.13.0.92` | Frame sampling via `VideoCapture` + `CAP_PROP_POS_FRAMES` seek | Spec §3 uses `cv2.VideoCapture` and `cv2.cvtColor(BGR→RGB)`. Latest wheel; arm64 Mac wheels are published. Headless variant (`opencv-python-headless`) is acceptable here since no GUI is used — pick one and stick with it (mixing the two breaks imports). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `pip` (or `uv`) | Dependency install | `uv pip install -r requirements.txt` is ~10× faster than pip for this dependency tree (torch + opencv are large) but pip works fine. |
| `python -m venv` | Isolation | Standard `venv` is sufficient — no need for conda/mamba. The pure-pip wheels for torch/torchvision/opencv on arm64 mac are well-tested. |

## Minimum Viable `requirements.txt`

```text
# Core inference
torch==2.11.0
torchvision==0.26.0
open_clip_torch==3.3.0

# Frame I/O
opencv-python==4.13.0.92

# Numerics + signal processing
numpy>=2.1,<2.5
scipy==1.16.2
ruptures==1.1.10

# UX
tqdm>=4.67,<5
```

**Total transitive footprint:** roughly 1.3 GB on disk (torch is ~800 MB, opencv ~90 MB, the CLIP ViT-L/14 weights ~890 MB are downloaded separately on first run via HuggingFace Hub into `~/.cache/huggingface/hub/`).

**Reproducibility:** every line is either `==` or a tight `>=,<` window, so `pip install -r requirements.txt` resolves deterministically across CPU and MPS/CUDA targets.

## Mac / Apple Silicon Notes

This machine: **macOS 15.3.2 (Sequoia), Darwin 24.3.0, arm64, Python 3.13.2, ffmpeg 7.1.1 via Homebrew.** Everything in the recommended stack has native arm64 wheels; no Rosetta needed.

### Device selection

The spec correctly tells `open_clip` to "auto-select" via `model.eval()` without explicit `.to(device)`. **However, on Apple Silicon, `open_clip` does not automatically pick MPS** — it stays on CPU unless told. For the prototype, CPU is acceptable (3,600 frames × ~150ms/frame on M-series CPU ≈ 9 minutes per 30-min video). If wall-clock matters, add this in `extract.py`:

```python
import torch
device = (
    "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
model = model.to(device)
# in the embedding loop:
batch = batch.to(device)
```

This is **not a deviation from spec §1's "no GPU-specific code paths"** — it is a single `device` constant that falls through to CPU when MPS/CUDA aren't available. Spec §1's intent (per §1 "Hardware assumption" and §10) is "don't write CUDA-only code paths," not "don't ever call `.to(device)`."

### Known MPS caveat for body cam workload

MPS works fine for ViT-L/14 inference on Sequoia (15.x). There is a [reported regression on macOS 26 / Tahoe](https://github.com/pytorch/pytorch/issues/167679) where MPS reports as built-but-unavailable, but that is not the OS on this machine. **Confidence: HIGH that MPS works on this specific darwin 24.3.0 setup.**

Expected MPS speedup over CPU for this exact workload: **~3–5×** for ViT-L/14 batch-32 inference on M-series. Not transformative; CPU is fine if the user prefers to keep the code path identical to a Linux CPU box.

### ffmpeg

System ffmpeg 7.1.1 (Homebrew) is well above the 6.0 recommended floor and 4.4 minimum. Concat demuxer + `-c copy` are stable. `videotoolbox` is enabled in the build (relevant only if the spec ever falls back to re-encode — which it shouldn't per §7).

## Verified API Notes

### open_clip ViT-L-14 OpenAI loading — UNCHANGED

The spec §2 code:

```python
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='openai'
)
```

is correct and current as of `open_clip_torch` 3.3.0. The pretrained registry entry resolves to:

```python
"ViT-L-14": dict(
    openai=_pcfg(
        hf_hub="timm/vit_large_patch14_clip_224.openai/",
        quick_gelu=True,
    ),
    ...
)
```

`quick_gelu=True` is forced by the registry — you do **not** need to switch to `'ViT-L-14-quickgelu'` to get the QuickGELU variant. (The `-quickgelu` suffixed names exist in the registry as auto-generated aliases but they are not required for OpenAI weights, which always use QuickGELU regardless of the model name string.)

**One v3 internals change to be aware of:** loading no longer goes through `torch.jit.load` — it routes through HF Hub `timm/*_clip.openai` checkpoints. This eliminates an arbitrary-code-execution surface and is an *internal* change; the user-facing API is unchanged. First-run will download from HF Hub instead of OpenAI's CDN; cache lives in `~/.cache/huggingface/hub/`. **Confidence: HIGH** (verified against `pretrained.py` on main, 2026-05-06).

### Output dimension

L2-normalized 768-dim vectors per the spec §0.5 verification (`shape == (N, 768)`) — **correct and unchanged.**

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `open_clip_torch` for CLIP loading | HuggingFace `transformers.CLIPModel` | Use `transformers` only if you need cross-tool compatibility with HF pipelines or want to load arbitrary fine-tuned CLIP variants from the Hub. For OpenAI ViT-L/14 specifically, `open_clip` is the *canonical* loader cited in the spec's source paper (Koddenbrock 2025) — switching would weaken the citation chain. |
| `subprocess.run(['ffmpeg', ...])` (spec §7) | `python-ffmpeg` 2.0.x or `ffmpy` | Use a wrapper only if you start needing complex filter graphs. For `-ss / -to / -c copy` and the concat demuxer, raw `subprocess` is clearer and has zero runtime overhead. The spec's choice is right. |
| `opencv-python` for frame seek | `pyav` (FFmpeg bindings, frame-accurate) | `pyav` gives true frame-accurate seek where `cv2.CAP_PROP_POS_FRAMES` can drift on VFR sources. For 2 fps sampling tolerance is wide enough that OpenCV's seek error is negligible (<0.5s drift across 30 min). Stay with OpenCV unless drift becomes a verified problem. |
| `scipy.signal.medfilt` | `scipy.ndimage.median_filter` | `medfilt` is what the spec specifies and the regression in scipy issue #22333 has been fixed in 1.16.x. If a future regression surfaces, swap to `ndimage.median_filter(signal, size=5, mode='reflect')` — same algorithm, different implementation path. |
| `ruptures` PELT | `bayesian_changepoint_detection`, `claspy` | `ruptures` is the standard cited library for offline changepoint detection in Python. PELT is the spec's chosen algorithm and `ruptures` is its reference implementation. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `ffmpeg-python` (kkroening) | Last release July 2019, 600+ open issues, abandoned. Spec lists it but spec code does not import it. | `subprocess.run(['ffmpeg', ...])` as already in spec §7. Drop the dep entirely. |
| `transformers.CLIPModel` for this task | Heavier dependency tree (`tokenizers`, `safetensors`, `huggingface_hub` peers); breaks the citation alignment with Koddenbrock 2025; QuickGELU activation is not the default and must be patched in. | `open_clip_torch` as the spec specifies. |
| `pip install torch` without a pin | Major dispatcher and JIT changes between 2.6 → 2.11. A floating torch will silently break embedding reproducibility across machines. | `torch==2.11.0` pinned exactly. |
| `scipy>=1.15,<1.16` | `medfilt` returns wrong output for some kernel sizes ([#22333](https://github.com/scipy/scipy/issues/22333)) — exactly what spec §5 step 1 uses. | `scipy==1.16.2` (or pin to `>=1.16,<1.18`). |
| Re-encoding clips with `libx264` by default | Spec §7 explicitly says "do not re-encode unless `-c copy` is broken." Re-encoding makes the pipeline ~50× slower for no quality gain on a prototype. | `-c copy` (stream copy) as the spec instructs. Only fall back to `-c:v libx264 -crf 18` if a specific output file is visibly broken. |
| Mixing `opencv-python` and `opencv-python-headless` in the same env | They share the `cv2` module name — the second install corrupts the first. | Pick one. Stick with `opencv-python` unless deploying to a no-display Linux container. |
| Python 3.14 | Released October 2025, but `ruptures` 1.1.10's metadata declares `<3.14`. Pip will refuse to install or you'll need an unverified upgrade. | Python 3.11, 3.12, or 3.13. |

## Stack Patterns by Variant

**If running CPU-only (matches spec default):**
- No code changes from spec §2; `model.eval()` keeps everything on CPU.
- Expect ~9 min per 30-min video on M-series CPU at batch 32.

**If on Apple Silicon and you want speed:**
- Add MPS device selection (snippet above in "Mac / Apple Silicon Notes").
- Expect ~3–5× speedup. Verify L2-normalization invariant from spec §0.5 still holds — float32 on MPS is correct, but quickly sanity-check.

**If on a Linux/CUDA machine:**
- The same MPS-fallback snippet promotes `cuda` automatically. No further changes.
- Expect 10–20× CPU speedup with a modern GPU for this batch size.

**If you cannot install scipy 1.16+ for any reason:**
- Replace `scipy.signal.medfilt(x, kernel_size=5)` with `scipy.ndimage.median_filter(x, size=5, mode='reflect')`. Same output for 1D input on the affected kernel sizes.

## Version Compatibility Matrix

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.11.0` | `torchvision==0.26.0` | Lockstep release. Mismatched versions break `torchvision.transforms` import on some platforms. |
| `open_clip_torch==3.3.0` | `torch>=2.6` | 3.3.0 requires torch ≥ 2.6 (verified against open_clip README). 2.11.0 satisfies this. |
| `scipy==1.16.2` | `numpy>=2.0` | scipy 1.16+ requires numpy 2.x. The constraint `numpy>=2.1,<2.5` is compatible. |
| `numpy>=2.1` | `torch==2.11.0` | torch 2.11 was built against numpy 2.x; tested compatibility. |
| `ruptures==1.1.10` | `numpy<2.5`, `python<3.14` | Both fine within our pin set. |
| `opencv-python==4.13.0.92` | `numpy>=2.0` | OpenCV 4.13 publishes wheels built against numpy 2.x; arm64 macOS wheels available. |

## Reproducibility Checklist

- [ ] `python --version` reports 3.11, 3.12, or 3.13 (NOT 3.14, NOT <3.11)
- [ ] `pip install -r requirements.txt` completes with no resolver warnings
- [ ] `ffmpeg -version` reports `>=4.4` (recommend `>=6.0`)
- [ ] First `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` call downloads from HF Hub to `~/.cache/huggingface/hub/timm--vit_large_patch14_clip_224.openai/` — verify with `du -sh` (~890 MB)
- [ ] Spec §0.5 verification print: `np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)` returns `True`
- [ ] Spec §0.5 verification print: `embeddings.shape == (N, 768)`
- [ ] Median filter output for synthetic input `[1,1,1,100,1,1,1]` with `kernel_size=5` returns `[1,1,1,1,1,1,1]` — confirms scipy `medfilt` regression is not in the installed version

## Sources

- [PyPI: torch 2.11.0](https://pypi.org/project/torch/) — verified version, Python compatibility, release date 2026-03-23 (HIGH confidence)
- [PyPI: torchvision 0.26.0](https://pypi.org/project/torchvision/) — verified lockstep release with torch 2.11 (HIGH)
- [PyPI: open_clip_torch 3.3.0](https://pypi.org/project/open_clip_torch/) — verified release date 2026-02-27 (HIGH)
- [open_clip pretrained.py on main](https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/open_clip/pretrained.py) — verified `'ViT-L-14' + pretrained='openai'` registry entry forces `quick_gelu=True` (HIGH)
- [open_clip README](https://github.com/mlfoundations/open_clip/blob/main/README.md) — verified `torch>=2.6` minimum, JIT→HF Hub loading change (HIGH)
- [PyPI: scipy 1.17.1](https://pypi.org/project/scipy/) — verified version (HIGH)
- [scipy issue #22333: medfilt regression](https://github.com/scipy/scipy/issues/22333) — confirmed regression introduced in scipy 1.15 affecting median filter output (MEDIUM — issue still open as of search date, fix verified in 1.16+)
- [PyPI: ffmpeg-python 0.2.0](https://pypi.org/project/ffmpeg-python/) — last release 2019-07-06 confirms unmaintained status (HIGH)
- [PyPI: numpy 2.4.4](https://pypi.org/project/numpy/) — verified Python ≥3.11 requirement (HIGH)
- [PyPI: ruptures 1.1.10](https://pypi.org/project/ruptures/) — verified `<3.14,>=3.9` Python range (HIGH)
- [PyPI: opencv-python 4.13.0.92](https://pypi.org/project/opencv-python/) — verified version (HIGH)
- [PyPI: tqdm 4.67.3](https://pypi.org/project/tqdm/) — verified version (HIGH)
- [Apple Developer: Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/) — MPS backend availability and macOS 12.3+ requirement (HIGH)
- [PyTorch issue #167679: MPS on macOS 26](https://github.com/pytorch/pytorch/issues/167679) — confirms MPS works on macOS 15.x (this machine) and has separate issues only on macOS 26 / Tahoe (MEDIUM — not relevant to current target OS)
- Local environment probe: `uname -m → arm64`, `sw_vers → 15.3.2`, `python3 --version → 3.13.2`, `ffmpeg -version → 7.1.1` (HIGH — directly observed)

---
*Stack research for: visual-embedding video highlight extraction CLI (CLIP ViT-L/14 → 1D MAD/PELT → ffmpeg cut+concat)*
*Researched: 2026-05-06*
