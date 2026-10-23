"""HTTP API cho frontend TypeScript."""

from __future__ import annotations

import base64
import io
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|\\[^A-Za-z\s]|[^\s]")

app = FastAPI(title="MathOCR API", version="1.0")


@lru_cache(maxsize=1)
def get_model():
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from unimernet.models.unimernet.unimernet import UniMERModel
    from unimernet.processors import load_processor

    model_dir = Path(snapshot_download("wanderkid/unimernet_tiny"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniMERModel.from_config(
        OmegaConf.create(
            {
                "model_name": "unimernet",
                "model_config": {"model_name": str(model_dir), "max_seq_len": 1536},
                "tokenizer_name": "nougat",
                "tokenizer_config": {"path": str(model_dir)},
                "load_pretrained": True,
                "pretrained": str(model_dir / "unimernet_tiny.pth"),
            }
        )
    ).to(device).eval()
    processor = load_processor(
        "formula_image_eval", OmegaConf.create({"image_size": [192, 672]})
    )

    def recognize(image: Image.Image) -> tuple[str, Image.Image] | None:
        """Tra ve LaTeX va dung canvas 672x192 ma model nhin thay, None neu anh hong."""
        canvas = processor.prepare_input(image)
        if canvas is None:
            return None
        tensor = processor.transform(image=np.array(canvas))["image"][:1].unsqueeze(0).to(device)
        return model.generate({"image": tensor})["pred_str"][0], canvas

    return recognize


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/webp"}:
        raise HTTPException(415, "Chi chap nhan PNG, JPEG, BMP hoac WebP")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Anh vuot qua 10 MB")
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "Tep khong phai anh hop le") from error
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(413, "Anh co qua nhieu pixel")

    model = get_model()
    started = time.perf_counter()
    prediction = model(image.convert("RGB"))
    latency_ms = (time.perf_counter() - started) * 1000
    if prediction is None:
        raise HTTPException(400, "Khong doc duoc noi dung anh")
    latex, canvas = prediction
    latex = str(latex).strip()

    buffer = io.BytesIO()
    canvas.convert("L").save(buffer, format="PNG")
    return {
        "latex": latex,
        "tokens": TOKEN_PATTERN.findall(latex),
        "latency_ms": round(latency_ms, 2),
        "source_size": [image.width, image.height],
        "image_size": [canvas.width, canvas.height],
        "input_image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


dist = ROOT / "web" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")

