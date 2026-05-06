"""
AksaraGuard AI — Gemini Relay on Modal (US Region)
====================================================
Transparent proxy — forwards Gemini API requests from restricted regions
(Indonesia, SEA) through Modal's US infrastructure.

URL format (same as Gemini API):
  POST /v1beta/models/{model}:generateContent?key={key}
  Body: standard Gemini generateContent JSON

Deploy: modal deploy gemini_relay.py
Then set GEMINI_PROXY_URL to the deployed URL.
"""
import modal

app = modal.App("aksaraguard-gemini-relay")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.115",
    "httpx>=0.27",
)


@app.function(image=image, scaledown_window=60)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "engine": "gemini-relay", "region": "us"}


@app.function(image=image, scaledown_window=120)
@modal.fastapi_endpoint(method="POST", path="/v1beta/models/{model}:generateContent")
async def generate_content(model: str, request: "fastapi.Request"):
    """
    Transparent proxy for Gemini generateContent.
    Forwards the request body + API key to actual Gemini API.
    """
    import httpx

    key = request.query_params.get("key", "")
    body = await request.json()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    print(f"[Relay] → Gemini: models/{model}:generateContent")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, params={"key": key}, json=body)
        print(f"[Relay] ← Gemini: HTTP {resp.status_code}")
        return resp.json()
