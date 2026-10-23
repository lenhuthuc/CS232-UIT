import io

from fastapi.testclient import TestClient
from PIL import Image

from src import api
from src.api import app


client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_rejects_non_image():
    response = client.post("/api/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_predict_uses_unimernet(monkeypatch):
    image = Image.new("RGB", (20, 10), "white")
    content = io.BytesIO()
    image.save(content, "PNG")
    canvas = Image.new("RGB", (672, 192), "white")
    monkeypatch.setattr(api, "get_model", lambda: lambda _: (r"\frac{x^2}{y+1}", canvas))
    response = client.post("/api/predict", files={"file": ("formula.png", content.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["latex"] == r"\frac{x^2}{y+1}"
    assert body["tokens"] == [r"\frac", "{", "x", "^", "2", "}", "{", "y", "+", "1", "}"]
    assert body["image_size"] == [672, 192]
    assert body["source_size"] == [20, 10]
    assert body["input_image"].startswith("data:image/png;base64,")


def test_predict_rejects_unreadable_image(monkeypatch):
    image = Image.new("RGB", (20, 10), "white")
    content = io.BytesIO()
    image.save(content, "PNG")
    monkeypatch.setattr(api, "get_model", lambda: lambda _: None)
    response = client.post("/api/predict", files={"file": ("formula.png", content.getvalue(), "image/png")})
    assert response.status_code == 400

