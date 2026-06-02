"""
Unified Bonsai Image server.

GPU (CUDA):  loads gemlite ternary pipeline from demo/models/ — fast.
CPU (no GPU): downloads prism-ml/bonsai-image-ternary-4B-unpacked (~10 GB)
              and runs with diffusers on CPU — works but slow (~30-90 min/image).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_DIR    = Path(__file__).parent
DEMO_DIR    = REPO_DIR / "demo"
VENDOR_DIR  = DEMO_DIR / "vendor" / "image-studio"
MODELS_DIR  = DEMO_DIR / "models"
GENERATED_DIR = REPO_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

# Make backend_gpu importable (deferred — gemlite imports inside functions)
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

CUDA = torch.cuda.is_available()

TERNARY_GEMLITE_DIR  = MODELS_DIR / "bonsai-image-4B-ternary-gemlite"
CPU_MODEL_LOCAL_DIR  = MODELS_DIR / "bonsai-image-4B-ternary-unpacked"
HF_CPU_MODEL_ID      = "prism-ml/bonsai-image-ternary-4B-unpacked"

_executor = ThreadPoolExecutor(max_workers=1)
_pipeline = None
_mode: str = "loading"   # "loading" | "gpu" | "cpu" | "error"
_mode_error: str = ""
_gpu_backend: str = ""


# ── loaders ──────────────────────────────────────────────────────────────────

BINARY_GEMLITE_DIR = MODELS_DIR / "bonsai-image-4B-binary-gemlite"


def _load_gpu_pipeline():
    """Load transformer + VAE on GPU, text encoder on CPU.

    Text encoder runs once per image — CPU speed is fine.
    Keeping it off-GPU frees ~2 GB VRAM for the transformer and
    its activation buffers, making the pipeline fit on 4 GB cards.
    """
    from backend_gpu.pipeline_gpu import (
        _load_gemlite_transformer, _load_vae, _load_scheduler,
    )
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.utils.patching import prepare_for_inference
    from transformers import AutoTokenizer

    # pick variant -------------------------------------------------------
    ternary_transformer = next(TERNARY_GEMLITE_DIR.glob("transformer-gemlite-*"), None)
    binary_transformer  = next(BINARY_GEMLITE_DIR.glob("transformer-gemlite-*"), None) \
        if BINARY_GEMLITE_DIR.is_dir() else None

    variant = os.environ.get("BONSAI_VARIANT", "auto")
    if variant == "auto":
        variant = "binary" if binary_transformer else "ternary"

    if variant == "binary":
        if binary_transformer is None:
            raise RuntimeError(
                "Binary model not found. Run:  cd demo && bash scripts/download_model.sh binary"
            )
        transformer_dir = binary_transformer
        model_dir       = BINARY_GEMLITE_DIR
    else:
        if ternary_transformer is None:
            raise RuntimeError(
                "Ternary model not found. Run:  cd demo && bash scripts/download_model.sh ternary"
            )
        transformer_dir = ternary_transformer
        model_dir       = TERNARY_GEMLITE_DIR

    log.info("GPU variant: %s  transformer: %s", variant, transformer_dir)

    # transformer on GPU (gemlite) ----------------------------------------
    transformer = _load_gemlite_transformer(transformer_dir, device="cuda:0")
    log.info("transformer on GPU — free VRAM: %.0f MB",
             torch.cuda.mem_get_info()[0] / 1024**2)

    # text encoder on CPU (HQQ pytorch — avoids ~2 GB GPU reservation) ---
    log.info("loading text encoder on CPU (HQQ pytorch)...")
    text_encoder = AutoHQQHFModel.from_quantized(
        str(model_dir / "text_encoder-hqq-4bit"),
        compute_dtype=torch.bfloat16,
        device="cpu",
    )
    prepare_for_inference(text_encoder, backend="pytorch")
    log.info("text encoder on CPU")

    # VAE on GPU ----------------------------------------------------------
    vae = _load_vae(model_dir / "vae", device="cuda:0")
    log.info("VAE on GPU — free VRAM: %.0f MB",
             torch.cuda.mem_get_info()[0] / 1024**2)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "text_encoder-hqq-4bit" / "tokenizer")
    )
    scheduler = _load_scheduler(transformer_dir)

    return _CpuComponents(transformer, text_encoder, tokenizer, vae, scheduler)


def _ensure_cpu_model():
    marker = CPU_MODEL_LOCAL_DIR / ".download_complete"
    if marker.exists():
        log.info("CPU model cache OK at %s", CPU_MODEL_LOCAL_DIR)
        return
    log.info("Downloading %s (~10 GB) — this takes a while...", HF_CPU_MODEL_ID)
    CPU_MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_CPU_MODEL_ID,
        local_dir=str(CPU_MODEL_LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    marker.touch()
    log.info("CPU model ready at %s", CPU_MODEL_LOCAL_DIR)


class _CpuComponents:
    """Holds individually-loaded model components for CPU inference."""
    def __init__(self, transformer, text_encoder, tokenizer, vae, scheduler):
        self.transformer  = transformer
        self.text_encoder = text_encoder
        self.tokenizer    = tokenizer
        self.vae          = vae
        self.scheduler    = scheduler


def _load_cpu_pipeline():
    _ensure_cpu_model()
    log.info("Loading CPU model components (bfloat16)...")
    base = CPU_MODEL_LOCAL_DIR

    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, AutoModelForCausalLM

    transformer = Flux2Transformer2DModel.from_pretrained(
        str(base / "transformer"), torch_dtype=torch.bfloat16
    ).to("cpu").eval()
    # diffusion_klein defaults _inference_dtype to float16 when unset;
    # CPU weights are bfloat16, so override to avoid dtype mismatch.
    transformer._inference_dtype = torch.bfloat16
    log.info("transformer loaded")

    vae = AutoencoderKLFlux2.from_pretrained(
        str(base / "vae"), torch_dtype=torch.bfloat16
    ).to("cpu").eval()
    log.info("vae loaded")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(base / "scheduler"))
    log.info("scheduler loaded")

    tokenizer = AutoTokenizer.from_pretrained(str(base / "tokenizer"))
    log.info("tokenizer loaded")

    text_encoder = AutoModelForCausalLM.from_pretrained(
        str(base / "text_encoder"), torch_dtype=torch.bfloat16
    ).to("cpu").eval()
    log.info("text_encoder loaded")

    log.info("CPU components ready")
    return _CpuComponents(transformer, text_encoder, tokenizer, vae, scheduler)


def _startup_load():
    global _pipeline, _mode, _mode_error, _gpu_backend
    try:
        if CUDA:
            variant = os.environ.get("BONSAI_VARIANT", "auto")
            log.info("CUDA detected — loading GPU pipeline (variant=%s)", variant)
            _pipeline = _load_gpu_pipeline()
            _gpu_backend = os.environ.get("BONSAI_VARIANT", "auto")
            _mode = "gpu"
            log.info("GPU pipeline ready")
        else:
            log.warning("No CUDA GPU — loading CPU pipeline (generation will be slow)")
            _pipeline = _load_cpu_pipeline()
            _mode = "cpu"
    except Exception as exc:
        log.exception("Pipeline load failed")
        _mode = "error"
        _mode_error = str(exc)


# ── inference ─────────────────────────────────────────────────────────────────

def _run_generate(prompt: str, seed: int, steps: int, width: int, height: int) -> bytes:
    # Both GPU and CPU paths return _CpuComponents and use diffusion_forward.
    # On GPU: transformer + VAE on cuda:0, text encoder on CPU.
    # On CPU: everything on CPU.
    from backend_gpu import diffusion_klein
    c = _pipeline
    image = diffusion_klein.diffusion_forward(
        transformer=c.transformer,
        text_encoder=c.text_encoder,
        tokenizer=c.tokenizer,
        vae=c.vae,
        scheduler=c.scheduler,
        prompt=prompt,
        height=height,
        width=width,
        num_steps=steps,
        seed=seed,
        guidance=1.0,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _startup_load)


class GenerateRequest(BaseModel):
    prompt: str
    seed: int | None = None
    steps: int = 4
    width: int = 1024
    height: int = 1024


@app.get("/status")
async def status():
    return {"mode": _mode, "cuda": CUDA, "error": _mode_error, "backend": _gpu_backend}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if _mode == "loading":
        raise HTTPException(503, "Pipeline still loading — try again in a moment")
    if _mode == "error":
        raise HTTPException(500, f"Pipeline failed to load: {_mode_error}")

    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    loop = asyncio.get_running_loop()
    try:
        png_bytes = await loop.run_in_executor(
            _executor,
            lambda: _run_generate(req.prompt, seed, req.steps, req.width, req.height),
        )
    except Exception as exc:
        log.exception("generate failed")
        raise HTTPException(500, str(exc))

    filename = f"{int(time.time())}_{seed}.png"
    (GENERATED_DIR / filename).write_bytes(png_bytes)
    return {"filename": filename, "seed": seed}


@app.get("/images")
async def list_images():
    files = sorted(GENERATED_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"filename": f.name} for f in files]


@app.get("/images/{filename}")
async def get_image(filename: str):
    path = GENERATED_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Not found")
    path.resolve().relative_to(GENERATED_DIR.resolve())
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=str(REPO_DIR / "static"), html=True), name="static")
