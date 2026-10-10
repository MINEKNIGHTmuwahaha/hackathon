"""
src/gemini_vision.py
--------------------
Gemini Vision wrapper for ECG image analysis.

Pipeline:
    photo of ECG → Gemini Vision → structured JSON →
    AAMI classes + confidence + referral decision

Uses the new google-genai SDK with AQ.-prefixed keys.
"""

import os
import json
from pathlib import Path
from typing import Optional

from .logger import get_logger


log = get_logger("sentinel.gemini")

MODEL_NAME = "gemini-flash-latest"

PROMPT = """
You are an ECG analysis assistant for a triage tool used in rural clinics.
Analyze the provided ECG image carefully.

Return ONLY valid JSON with these fields, no preamble, no markdown fences:

{
  "image_quality": "good" | "acceptable" | "poor",
  "is_ecg": true | false,
  "leads_visible": ["II", "V1", ...],
  "heart_rate_bpm": number | null,
  "rhythm": "sinus" | "irregular" | "afib" | "paced" | "unknown",
  "beats": [
    {
      "index": 1,
      "morphology": "normal" | "narrow_irregular" | "wide_bizarre" | "paced" | "unclear",
      "aami_class_guess": "N" | "S" | "V" | "F" | "Q" | "?",
      "confidence": 0.0
    }
  ],
  "annotations_detected": [],
  "red_flags": [],
  "referral_recommended": true | false,
  "referral_reason": "short reason if true"
}

Rules:
- If the image is NOT an ECG, set is_ecg=false and stop.
- Never diagnose a specific disease. Only triage.
- If uncertain about a beat, use "?" and lower confidence.
- Be conservative: when in doubt, recommend referral.
- AAMI classes: N (normal), S (supraventricular), V (ventricular),
  F (fusion), Q (unknown/paced).
"""


def _get_client():
    """Lazy-load the Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: python -m pip install google-genai pillow"
        )

    key = os.environ.get("GEMINI_API_KEY")

    # Fallback: read from .env at project root
    if not key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Either:\n"
            "  1. Put it in .env at project root as: GEMINI_API_KEY=AQ...\n"
            "  2. Or run: setx GEMINI_API_KEY \"your-key\" "
            "and open a NEW terminal."
        )

    return genai.Client(api_key=key)


def _extract_json(text: str) -> dict:
    """Strip markdown fences if Gemini wraps the JSON."""
    text = text.strip()
    if text.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Gemini JSON: {e}")
        log.debug(f"Raw text: {text[:500]}")
        return {}


def analyze_ecg_image(image_path: str) -> dict:
    """
    Send an ECG image to Gemini Vision.

    Args:
        image_path: path to a .jpg / .png file

    Returns:
        {
            "ok": bool,
            "error": str | None,
            "result": {...} | None
        }
    """
    from PIL import Image

    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}", "result": None}

    try:
        client = _get_client()
    except Exception as e:
        return {"ok": False, "error": str(e), "result": None}

    try:
        img = Image.open(p)

        # Retry on 503 (server overloaded) and 429 (rate limit)
        import time
        max_attempts = 4
        last_error = None
        response = None

        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[PROMPT, img],
                )
                break  # success
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str \
                        or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                    log.warning(f"Gemini overloaded, retrying in {wait}s "
                                f"(attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait)
                    continue
                # Any other error — fail immediately
                raise

        if response is None:
            return {
                "ok": False,
                "error": f"Gemini unavailable after {max_attempts} tries: {last_error}",
                "result": None,
            }

        raw = response.text
        log.info(f"Gemini response: {len(raw)} chars")

        parsed = _extract_json(raw)
        if not parsed:
            return {
                "ok": False,
                "error": "Could not parse JSON from Gemini",
                "result": None,
                "raw": raw,
            }

        return {"ok": True, "error": None, "result": parsed, "raw": raw}

    except Exception as e:
        log.error(f"Gemini call failed: {e}")
        return {"ok": False, "error": str(e), "result": None}

   
    if result["ok"] and s1_signal is not None:
        try:
            from .digitizer.learn import record_pair
            # Extract a rough signal from Gemini's beat positions
            beats = result["result"].get("beats") or []
            beat_positions = [b.get("estimated_position_px") for b in beats]
            # Build a rough signal from beat positions (sparse)
            if beat_positions and all(p is not None for p in beat_positions):
                n = len(s1_signal)
                s2_signal = np.zeros(n)
                for i, pos in enumerate(beat_positions):
                    if isinstance(pos, list) and len(pos) == 2:
                        x = int(pos[0] * n / 1000)  # normalize px to sample idx
                        x = min(max(x, 0), n - 1)
                        s2_signal[x] = 1.0
                record_pair(image_path, s1_signal, s2_signal, s1_quality)
        except Exception as e:
            log.warning(f"learn hook failed: {e}")

    return result

def summarize_for_ui(gemini_result: dict) -> dict:
    """
    Convert the raw Gemini result into UI-friendly metrics.

    Returns:
        {
            "n_beats": int,
            "class_counts": {"N": int, "S": int, ...},
            "referral_count": int,
            "referral_pct": float,
            "referral_recommended": bool,
            "referral_reason": str,
            "heart_rate_bpm": int | None,
            "rhythm": str,
            "image_quality": str,
            "red_flags": list[str],
        }
    """
    beats = gemini_result.get("beats") or []
    n = len(beats)

    counts = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
    ref = 0
    DANGER = {"V", "S", "F", "Q"}

    for b in beats:
        cls = b.get("aami_class_guess", "?")
        conf = float(b.get("confidence", 0.0))
        if cls in counts:
            counts[cls] += 1
        if cls in DANGER or conf < 0.70:
            ref += 1

    ref_pct = (100.0 * ref / n) if n > 0 else 0.0

    return {
        "n_beats": n,
        "class_counts": counts,
        "referral_count": ref,
        "referral_pct": ref_pct,
        "referral_recommended": gemini_result.get("referral_recommended", False),
        "referral_reason": gemini_result.get("referral_reason", ""),
        "heart_rate_bpm": gemini_result.get("heart_rate_bpm"),
        "rhythm": gemini_result.get("rhythm", "unknown"),
        "image_quality": gemini_result.get("image_quality", "unknown"),
        "red_flags": gemini_result.get("red_flags") or [],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.gemini_vision <path-to-ecg-image>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"Analyzing {img_path}...")
    out = analyze_ecg_image(img_path)

    if not out["ok"]:
        print(f"ERROR: {out['error']}")
        sys.exit(1)

    print("\n--- raw Gemini output ---")
    print(json.dumps(out["result"], indent=2))

    print("\n--- UI summary ---")
    summary = summarize_for_ui(out["result"])
    print(json.dumps(summary, indent=2, default=str))