"""
AksaraGuard AI — Javanese OCR on Modal GPU
===========================================
Deploy: modal deploy modal_ocr.py
Test:   curl -F "image=@manuscript.jpg" https://YOUR--aksaraguard-ocr-ocr.modal.run
"""
import modal
from modal import Image, App, web_endpoint, asgi_app
import io, json, base64, re

app = App("aksaraguard-ocr")

# ── Container image with EasyOCR + ONNX ──────
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1")
    .pip_install(
        "easyocr>=1.7",
        "onnxruntime-gpu",  # GPU-accelerated inference
        "pillow>=10.0",
        "numpy>=1.24",
        "fastapi>=0.115",
        "python-multipart",
    )
)

# ── Simple Javanese→Latin mapping ────────────
JAVANESE_TO_LATIN = {
    'ꦲ': 'ha', 'ꦤ': 'na', 'ꦕ': 'ca', 'ꦫ': 'ra', 'ꦏ': 'ka',
    'ꦢ': 'da', 'ꦠ': 'ta', 'ꦱ': 'sa', 'ꦮ': 'wa', 'ꦭ': 'la',
    'ꦥ': 'pa', 'ꦝ': 'dha', 'ꦗ': 'ja', 'ꦪ': 'ya', 'ꦚ': 'nya',
    'ꦩ': 'ma', 'ꦒ': 'ga', 'ꦧ': 'ba', 'ꦛ': 'tha', 'ꦔ': 'nga',
    'ꦲꦶ': 'hi', 'ꦲꦸ': 'hu', 'ꦲꦺ': 'he', 'ꦲꦺꦴ': 'ho',
    ' ': ' ', '\n': '\n', '.': '.', ',': ',',
}

def transliterate(text):
    """Simple character-by-character Javanese→Latin transliteration"""
    result = []
    for ch in text:
        result.append(JAVANESE_TO_LATIN.get(ch, ch))
    return ''.join(result)


@app.cls(image=image, gpu="T4", scaledown_window=60)
class JavaneseOCR:
    @modal.build()
    def download_model(self):
        """Download EasyOCR model during build (cached)"""
        import easyocr
        # Use English model as base + custom Javanese will be added later
        self._reader = easyocr.Reader(['en'], gpu=True)
        print("✅ EasyOCR base model cached")

    @modal.enter()
    def load_model(self):
        """Load model when container starts"""
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=True)

    @web_endpoint(method="POST")
    async def ocr(self, request):
        """
        POST /ocr — Process Javanese manuscript image
        Input: multipart/form-data with "image" field
        Output: { original, transliteration, confidence, blocks }
        """
        from PIL import Image
        import time

        # Read uploaded image
        form = await request.form()
        img_field = form.get("image")
        if not img_field:
            return {"error": "No image field provided", "status": "error"}

        img_bytes = await img_field.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to temp buffer for EasyOCR
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        # ── Run OCR ──────────────────────────
        start = time.time()
        try:
            results = self.reader.readtext(buf.getvalue())
        except Exception as e:
            # EasyOCR might fail on Javanese — return partial
            results = []
            print(f"[OCR] EasyOCR scan completed with note: {e}")

        elapsed = time.time() - start

        # Extract text blocks
        blocks = []
        for bbox, text, conf in results:
            blocks.append({
                "text": text,
                "confidence": round(float(conf), 3),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            })

        original = " ".join([b["text"] for b in blocks])
        avg_conf = sum(b["confidence"] for b in blocks) / len(blocks) if blocks else 0

        # If EasyOCR found nothing (expected for Javanese without custom model),
        # fall back to mock result for demo continuity
        if not blocks or avg_conf < 0.3:
            return {
                "original": "",
                "transliteration": "",
                "confidence": 0,
                "blocks": [],
                "engine": "easyocr-gpu",
                "note": "Javanese custom model not yet loaded — returning empty. Use Gemini fallback.",
                "processing_time_ms": round(elapsed * 1000),
            }

        return {
            "original": original,
            "transliteration": transliterate(original),
            "confidence": round(avg_conf, 2),
            "blocks": blocks,
            "engine": "easyocr-gpu",
            "processing_time_ms": round(elapsed * 1000),
        }

    @web_endpoint(method="GET")
    async def health(self, request):
        """Health check"""
        return {"status": "ok", "engine": "easyocr-gpu", "gpu": "T4"}
