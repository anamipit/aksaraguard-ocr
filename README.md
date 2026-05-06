# AksaraGuard OCR — Modal GPU Service

Javanese script OCR powered by EasyOCR running on Modal.com GPU (T4).

## 🚀 Deploy (2 menit)

```bash
# 1. Setup venv (Mac)
python3 -m venv ~/modal-env
source ~/modal-env/bin/activate
pip install modal

# 2. Login Modal (sekali aja)
modal token new

# 3. Deploy
modal deploy modal_ocr.py
```

Output:
```
✓ Created/Updated app 'aksaraguard-ocr'
✓ App deployed! 🎉
  https://YOUR_USER--aksaraguard-ocr-ocr.modal.run
```

Copy URL-nya → set `MODAL_OCR_URL` di backend AdonisJS.

## 🔧 Test

```bash
curl -F "image=@manuscript.jpg" https://YOUR_USER--aksaraguard-ocr-ocr.modal.run
```

## ⚙️ Tech Stack

- **EasyOCR** — deep learning OCR engine
- **ONNX Runtime GPU** — accelerated inference on T4
- **Modal.com** — serverless GPU cloud ($30/bulan gratis kredit)

## 📝 Custom Model (Fase 2)

Saat ini EasyOCR pakai English model sebagai base. Untuk Javanese, perlu training custom model dengan 2000+ gambar sintetis. Lihat `train/` directory (coming soon).
