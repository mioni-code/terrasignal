"""Optional Gemini integration (google-genai SDK). The app works without it."""
from __future__ import annotations

import os

import config

IMAGE_PROMPT = (
    "You are an environmental image analysis assistant. Describe only the visible "
    "features in the provided aerial photograph. Identify possible signs of "
    "environmental disturbance, vegetation damage, unusual surface patterns or visible "
    "waste. Distinguish direct visual observations from hypotheses. Do not claim to "
    "identify chemical substances, toxicity or contamination from visual appearance "
    "alone. State uncertainties and recommend what additional field measurements would "
    "be needed.")

REPORT_PROMPT = (
    "You are an environmental screening report writer. Using ONLY the data below, write "
    "a structured report with these numbered sections: 1. Executive summary. "
    "2. Detected changes. 3. Possible environmental significance. 4. Evidence available. "
    "5. Uncertainties. 6. Recommended next steps. 7. Additional data required. "
    "Clearly label index values as 'SATELLITE-DERIVED INDEX MEASUREMENT' and "
    "interpretations as 'AI-GENERATED HYPOTHESIS'. " + config.INDEX_DISCLAIMER + " "
    "Never describe NDVI or NDWI values as confirmed, field-verified or ground-truth "
    "measurements. An NDWI change is a change in a reflectance index; it is not by itself "
    "a confirmed change in water extent or water quality. Do not state or imply chemical "
    "pollution, toxicity or a specific cause. Do not invent numbers, costs or facts not "
    "present in the data. This is a preliminary screening report, not an official "
    "assessment.\n\nDATA:\n")


def api_key() -> str | None:
    return config.get_secret("GEMINI_API_KEY")


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def is_available() -> tuple[bool, str]:
    if not api_key():
        return False, "GEMINI_API_KEY not set"
    try:
        from google import genai  # noqa: F401
    except Exception:
        return False, "google-genai not installed"
    return True, "Gemini configured"


def _client():
    from google import genai
    return genai.Client(api_key=api_key())


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[str | None, str | None]:
    """Returns (text, error). Never raises."""
    ok, msg = is_available()
    if not ok:
        return None, msg
    try:
        from google.genai import types
        resp = _client().models.generate_content(
            model=model_name(),
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                      IMAGE_PROMPT])
        text = (resp.text or "").strip()
        return (text, None) if text else (None, "Gemini returned an empty response")
    except Exception as exc:
        return None, f"Gemini request failed ({type(exc).__name__})"


def generate_report(context: str) -> tuple[str | None, str | None]:
    """Returns (text, error). Never raises."""
    ok, msg = is_available()
    if not ok:
        return None, msg
    try:
        resp = _client().models.generate_content(
            model=model_name(), contents=REPORT_PROMPT + context)
        text = (resp.text or "").strip()
        return (text, None) if text else (None, "Gemini returned an empty response")
    except Exception as exc:
        return None, f"Gemini request failed ({type(exc).__name__})"
