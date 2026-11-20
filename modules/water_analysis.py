"""NDWI and water-surface extent change.

Two different things are reported and must not be conflated:
  * NDWI index change  - difference of a reflectance index (can reflect water level,
    wetness, turbidity, vegetation, sun-glint or residual cloud);
  * estimated open-water extent change - area where NDWI > 0 in one period but not the
    other (a threshold-based estimate, not a confirmed surface-water measurement).
NDWI does NOT detect chemical pollution or water quality.
"""
from __future__ import annotations

import numpy as np

WATER_THRESHOLD = 0.0  # NDWI > 0 is commonly used as an open-water indicator

NDWI_NOTE = ("NDWI is a reflectance-based water/wetness index. An NDWI change is not by "
             "itself a confirmed change in water extent, and NDWI cannot identify chemical "
             "pollution or water contamination.")


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (B03 - B08) / (B03 + B08); NaN where undefined."""
    green = green.astype("float32")
    nir = nir.astype("float32")
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (green - nir) / denom
    out[~np.isfinite(out) | (denom == 0)] = np.nan
    return out


def water_mask(ndwi: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return ndwi > WATER_THRESHOLD


def water_extent_change(baseline: np.ndarray, recent: np.ndarray,
                        pixel_area_m2: float) -> dict:
    """ESTIMATED open-water extent change (hectares) from the NDWI > 0 masks of two
    composites. This is separate from, and not implied by, an NDWI index change."""
    both = np.isfinite(baseline) & np.isfinite(recent)
    wb, wr = water_mask(baseline) & both, water_mask(recent) & both
    ha = pixel_area_m2 / 10_000
    return {
        "baseline_water_ha": float(wb.sum() * ha),
        "recent_water_ha": float(wr.sum() * ha),
        "lost_ha": float((wb & ~wr).sum() * ha),
        "gained_ha": float((~wb & wr).sum() * ha),
        "compared_ha": float(both.sum() * ha),
    }


def region_water_extent(baseline: np.ndarray, recent: np.ndarray, bbox_px: tuple,
                        region_mask: np.ndarray, pixel_area_m2: float) -> dict:
    """Same estimate restricted to the pixels of one detected region.

    bbox_px = (row0, row1, col0, col1); region_mask is the boolean region footprint
    inside that window."""
    r0, r1, c0, c1 = bbox_px
    b, r = baseline[r0:r1, c0:c1], recent[r0:r1, c0:c1]
    both = np.isfinite(b) & np.isfinite(r) & region_mask
    wb, wr = water_mask(b) & both, water_mask(r) & both
    ha = pixel_area_m2 / 10_000
    return {
        "baseline_water_ha": float(wb.sum() * ha),
        "recent_water_ha": float(wr.sum() * ha),
        "lost_ha": float((wb & ~wr).sum() * ha),
        "gained_ha": float((~wb & wr).sum() * ha),
    }
