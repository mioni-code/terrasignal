"""Central configuration for TerraSignal."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PREVIEW_DIR = DATA_DIR / "previews"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
DEMO_IMAGES_DIR = ASSETS_DIR / "demo_images"
DB_PATH = Path(os.getenv("TERRASIGNAL_DB", DATA_DIR / "terrasignal.db"))

for _d in (DATA_DIR, UPLOAD_DIR, PREVIEW_DIR, REPORTS_DIR, DEMO_IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_NAME = "TerraSignal"
TAGLINE = "Detect Early. Verify Precisely. Protect Nature."

# ---- Satellite -------------------------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_SCENES_PER_PERIOD = 4       # scenes composited per period
MAX_SEARCH_ITEMS = 30           # items fetched per STAC search
MAX_AOI_KM2 = 50.0              # larger AOIs are rejected (keeps reads small)
MIN_VALID_FRACTION = 0.30       # min share of valid pixels in a composite
# SCL classes treated as usable: 4 vegetation, 5 not vegetated, 6 water
SCL_VALID = (4, 5, 6)

DEFAULT_NDVI_THRESHOLD = 0.20   # drop in NDVI that counts as an anomaly
DEFAULT_NDWI_THRESHOLD = 0.30
DEFAULT_MIN_REGION_HA = 0.5

# ---- Domain lists ----------------------------------------------------------
CATEGORIES = ["Water bodies", "Riverbanks", "Agricultural land", "Forests",
              "Industrial facilities", "Other"]
FREQUENCIES = ["Weekly", "Bi-weekly", "Monthly", "Quarterly"]
INSPECTION_STATUSES = ["Pending", "Ready for review", "Analysed", "Report generated"]
PRIORITIES = ["Normal", "Attention", "High"]
PRIORITY_LABELS = {"Normal": "Normal change", "Attention": "Requires attention",
                   "High": "High priority"}

FREQUENCY_NOTE = ("Monitoring frequency is subject to satellite revisit schedules "
                  "(Sentinel-2 revisits a location roughly every 2–5 days), image "
                  "availability and weather conditions (cloud cover).")

# ---- Design ----------------------------------------------------------------
COLORS = {
    "primary": "#1F4D36",     # dark forest green
    "secondary": "#8FA98F",   # muted sage
    "bg": "#F6F8F5",
    "card": "#FFFFFF",
    "border": "#E1E7DF",
    "text": "#2B2F2C",        # charcoal
    "muted": "#6B756D",
    "High": "#B3261E",
    "Attention": "#D98E04",
    "Normal": "#5E8C61",
}

DISCLAIMER = ("PRELIMINARY ENVIRONMENTAL SCREENING REPORT. This is not an official "
              "environmental assessment, audit or legal document. Satellite indices "
              "(NDVI/NDWI) indicate changes in surface reflectance, not their cause. "
              "Priority scores are heuristic screening indicators, not probabilities. "
              "Field verification is required before any conclusion or action.")


INDEX_DISCLAIMER = ("NDVI and NDWI values are remotely sensed indicators derived from "
                    "satellite reflectance. They are not field-confirmed environmental "
                    "measurements.")


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, then from Streamlit secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:  # pragma: no cover - only inside Streamlit
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None
