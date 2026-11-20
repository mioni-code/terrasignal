"""Demonstration-only drone verification workflow.

NO drone is controlled, dispatched or navigated. The module only handles an
uploaded (or illustrative) aerial photograph and the inspection status.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

import config
import database as db

SIM_LABEL = "SIMULATED DRONE INSPECTION — NO REAL FLIGHT"
DEMO_IMAGE_NOTICE = ("This built-in image is ILLUSTRATIVE and synthetic. It is not a "
                     "photograph of the satellite anomaly or of any real location.")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_MB = 10
DEMO_IMAGE_PATH = config.DEMO_IMAGES_DIR / "illustrative_aerial.png"


def ensure_demo_image() -> Path:
    """Generate a synthetic aerial-style image (once)."""
    if DEMO_IMAGE_PATH.exists():
        return DEMO_IMAGE_PATH
    rng = np.random.default_rng(3)
    h, w = 540, 800
    base = np.zeros((h, w, 3), dtype="float32")
    base[..., 0], base[..., 1], base[..., 2] = 70, 120, 60
    noise = rng.normal(0, 10, (h, w, 1))
    base += noise
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    d.polygon([(260, 200), (430, 170), (520, 260), (470, 360), (300, 350)],
              fill=(150, 125, 85))        # bare/disturbed patch
    d.ellipse([560, 90, 700, 170], fill=(70, 100, 130))   # pond-like feature
    d.line([(0, 470), (800, 440)], fill=(190, 180, 160), width=14)  # track
    d.rectangle([0, h - 34, w, h], fill=(0, 0, 0))
    d.text((12, h - 26), "ILLUSTRATIVE SYNTHETIC IMAGE - NOT A REAL PHOTOGRAPH",
           fill=(255, 255, 255))
    img.save(DEMO_IMAGE_PATH)
    return DEMO_IMAGE_PATH


def validate_and_save_upload(filename: str, data: bytes) -> Path:
    """Validate an uploaded image and store it. Raises ValueError (friendly)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError("Unsupported file type. Please upload a JPG, PNG or WEBP image.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The image is larger than {MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("The file could not be read as an image.") from None
    img.thumbnail((2048, 2048))
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
    img.save(path, "JPEG", quality=90)
    return path


def attach_image(inspection_id: int, path: Path, is_demo_image: bool) -> None:
    db.update_inspection(inspection_id, uploaded_image_path=str(path),
                         image_is_demo=int(is_demo_image),
                         inspection_status="Ready for review")


def deterministic_verification_summary(insp: dict, anomaly: dict | None,
                                       image_is_demo: bool) -> str:
    """Used when Gemini is unavailable. Describes only what is known."""
    lines = [
        "VERIFICATION SUMMARY (rule-based; no AI image analysis performed)",
        f"Inspection: {insp.get('inspection_code')} at "
        f"{insp.get('latitude'):.5f}, {insp.get('longitude'):.5f}.",
    ]
    if anomaly:
        lines.append(f"Trigger: {anomaly['anomaly_type']}, approx. "
                     f"{anomaly.get('anomaly_area') or 0:.2f} ha, "
                     f"observed {anomaly.get('observation_date') or 'n/a'}.")
    if image_is_demo:
        lines.append("The attached image is an illustrative demonstration image and is "
                     "NOT evidence about this location.")
    else:
        lines.append("An aerial image was attached. It has not been machine-interpreted "
                     "(Gemini unavailable); manual review is required.")
    lines.append("This workflow is simulated. No drone flight took place. Field "
                 "measurements are needed to confirm any environmental condition.")
    return "\n".join(lines)
