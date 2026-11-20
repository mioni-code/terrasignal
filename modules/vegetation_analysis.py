"""NDVI, SCL cloud masking and temporal compositing."""
from __future__ import annotations

import numpy as np

import config


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """True where the Scene Classification Layer marks a usable pixel
    (vegetation, not-vegetated, water). Clouds, shadows, snow, defective and
    unclassified pixels are excluded."""
    return np.isin(scl, config.SCL_VALID)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (B08 - B04) / (B08 + B04); NaN where undefined."""
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (nir - red) / denom
    out[~np.isfinite(out) | (denom == 0)] = np.nan
    return out


def masked_index(index: np.ndarray, scl: np.ndarray) -> np.ndarray:
    out = index.copy()
    out[~scl_valid_mask(scl)] = np.nan
    return out


def composite(stack: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of the valid observations in a period."""
    if not stack:
        raise ValueError("Empty stack")
    arr = np.stack(stack).astype("float32")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(arr, axis=0)


def valid_fraction(a: np.ndarray, aoi_mask: np.ndarray | None = None) -> float:
    """Share of finite pixels; restricted to the AOI polygon when a mask is given."""
    if aoi_mask is not None:
        n = int(aoi_mask.sum())
        return float(np.isfinite(a[aoi_mask]).sum() / n) if n else 0.0
    return float(np.isfinite(a).mean()) if a.size else 0.0


def compute_change(recent: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """NDVI_change = NDVI_recent - NDVI_baseline."""
    return recent - baseline
