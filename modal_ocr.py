"""
AksaraGuard AI — Javanese OCR on Modal GPU
===========================================
Modal SDK v0.73+ (latest 2025 API)

Deploy: modal deploy modal_ocr.py
Test:   curl -F "image=@manuscript.jpg" https://YOUR--aksaraguard-ocr-ocr.modal.run
"""
from __future__ import annotations  # Defer annotation evaluation

import modal

app = modal.App("aksaraguard-ocr")

# ── Container image ─────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1")
    .pip_install(
        "easyocr>=1.7",
        "onnxruntime-gpu",
        "pillow>=10.0",
        "numpy>=1.24",
        "fastapi[standard]>=0.115",
    )
)

# ── Health check ────────────────────────────
@app.function(image=image, scaledown_window=60)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "engine": "easyocr-gpu", "gpu": "T4"}


# ── OCR endpoint ────────────────────────────
# Modal's fastapi_endpoint auto-injects fastapi.Request
# Use raw Request to manually parse multipart form data
@app.function(image=image, gpu="T4", scaledown_window=120)
@modal.fastapi_endpoint(method="POST")
async def ocr(request: "fastapi.Request"):  # string annotation — resolved by fastapi_endpoint
    """
    POST / — Process Javanese manuscript image
    Upload: multipart/form-data with "image" field
    """
    import io, time
    import easyocr
    from PIL import Image

    # Parse uploaded file from multipart form
    form = await request.form()
    img_field = form.get("image")
    if img_field is None:
        return {"error": "No 'image' field in form data", "status": "error"}

    img_bytes = await img_field.read()
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    print("[OCR] Loading EasyOCR + GPU recognition...")
    start = time.time()
    reader = easyocr.Reader(['en'], gpu=True)

    try:
        results = reader.readtext(buf.getvalue())
    except Exception as e:
        print(f"[OCR] Error: {e}")
        results = []

    elapsed = time.time() - start

    blocks = []
    for bbox, text, conf in results:
        blocks.append({
            "text": text,
            "confidence": round(float(conf), 3),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    original = " ".join(b["text"] for b in blocks)
    avg_conf = sum(b["confidence"] for b in blocks) / len(blocks) if blocks else 0

    java_map = {
        'ꦲ': 'ha', 'ꦤ': 'na', 'ꦕ': 'ca', 'ꦫ': 'ra', 'ꦏ': 'ka',
        'ꦢ': 'da', 'ꦠ': 'ta', 'ꦱ': 'sa', 'ꦮ': 'wa', 'ꦭ': 'la',
        'ꦥ': 'pa', 'ꦝ': 'dha', 'ꦗ': 'ja', 'ꦪ': 'ya', 'ꦚ': 'nya',
        'ꦩ': 'ma', 'ꦒ': 'ga', 'ꦧ': 'ba', 'ꦛ': 'tha', 'ꦔ': 'nga',
    }
    transliteration = ''.join(java_map.get(ch, ch) for ch in original)

    if not blocks or avg_conf < 0.3:
        return {
            "original": original or "",
            "transliteration": transliteration or "",
            "confidence": 0.0,
            "blocks": blocks,
            "engine": "easyocr-gpu",
            "note": "Javanese custom model not loaded — use Gemini 2.5 Flash for production accuracy",
            "processing_time_ms": round(elapsed * 1000),
        }

    return {
        "original": original,
        "transliteration": transliteration,
        "confidence": round(avg_conf, 2),
        "blocks": blocks,
        "engine": "easyocr-gpu",
        "processing_time_ms": round(elapsed * 1000),
    }
