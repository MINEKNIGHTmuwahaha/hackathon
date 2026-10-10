import time
from pathlib import Path
from google import genai
from PIL import Image

# Load key from .env
key = None
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            key = line.split("=", 1)[1].strip()
            break

if not key:
    raise SystemExit("GEMINI_API_KEY not found in .env")

print(f"key loaded: {key[:6]}...{key[-4:]} (len={len(key)})")

client = genai.Client(api_key=key)
img = Image.open("data/samples/test_ecg_v.jpg")

# First call — warms up
t0 = time.time()
r1 = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Is this an ECG? One word.", img],
)
t1 = time.time()
print(f"call 1: {t1-t0:.2f}s — {r1.text.strip()}")

# Second call — real speed
t0 = time.time()
r2 = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["How many beats are visible? Number only.", img],
)
t1 = time.time()
print(f"call 2: {t1-t0:.2f}s — {r2.text.strip()}")