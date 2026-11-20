"""Change detection: thresholding, connected regions, priority, orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np
from scipy import ndimage

import config
from modules import satellite_data as sat
from modules import vegetation_analysis as veg
from modules import water_analysis as wat

LonLat = Callable[[float, float], tuple[float, float]]  # (col, row) -> (lon, lat)


# ------------------------------------------------------------------ priority
def priority_score(mean_change: float, area_ha: float, threshold: float) -> float:
    """Heuristic 0-1 screening indicator (NOT a probability or AI confidence).
    60% magnitude of change relative to the threshold, 40% affected area."""
    severity = min(1.0, abs(mean_change) / (max(threshold, 0.05) * 3))
    size = min(1.0, area_ha / 5.0)
    return round(0.6 * severity + 0.4 * size, 3)


def priority_level(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Attention"
    return "Normal"


# ---------------------------------------------------------------- detection
def detect_regions(change: np.ndarray, threshold: float, direction: str,
                   pixel_area_m2: float, to_lonlat: LonLat,
                   min_area_ha: float = config.DEFAULT_MIN_REGION_HA,
                   max_regions: int = 40,
                   aoi_mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Group adjacent anomalous pixels into regions and describe them.

    direction: 'decrease' (change <= -threshold) or 'increase' (>= threshold).
    aoi_mask: optional boolean array (True = inside the user's polygon). Pixels outside
    it can never become part of a region.
    """
    if direction not in ("decrease", "increase"):
        raise ValueError("direction must be 'decrease' or 'increase'")
    thr = abs(threshold)
    with np.errstate(invalid="ignore"):
        mask = change <= -thr if direction == "decrease" else change >= thr
    mask &= np.isfinite(change)
    if aoi_mask is not None:
        if aoi_mask.shape != change.shape:
            raise ValueError("aoi_mask must have the same shape as the change array")
        mask &= aoi_mask
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labels, ids)
    slices = ndimage.find_objects(labels)
    out = []
    for lab, npx, sl in zip(ids, sizes, slices):
        area_ha = float(npx) * pixel_area_m2 / 10_000
        if area_ha < min_area_ha:
            continue  # suppress small, noisy regions
        region = labels[sl] == lab
        vals = change[sl][region]
        cy, cx = ndimage.center_of_mass(labels == lab)
        lon, lat = to_lonlat(cx + 0.5, cy + 0.5)
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        corners = [to_lonlat(c0, r0), to_lonlat(c1, r0), to_lonlat(c1, r1),
                   to_lonlat(c0, r1), to_lonlat(c0, r0)]
        mean = float(np.mean(vals))
        score = priority_score(mean, area_ha, thr)
        out.append({
            "latitude": float(lat), "longitude": float(lon),
            "area_ha": round(area_ha, 3), "mean_change": round(mean, 4),
            "peak_change": round(float(vals.min() if direction == "decrease"
                                       else vals.max()), 4),
            "pixels": int(npx), "direction": direction,
            "priority_score": score, "priority": priority_level(score),
            "polygon": {"type": "Polygon",
                        "coordinates": [[[float(x), float(y)] for x, y in corners]]},
            "bbox_px": (r0, r1, c0, c1),
            "mask_px": region,   # footprint inside bbox_px (in-memory only, not persisted)
        })
    out.sort(key=lambda d: d["priority_score"], reverse=True)
    return out[:max_regions]


# ------------------------------------------------------------- orchestration
@dataclass
class AnalysisResult:
    mode: str                                  # "REAL" or "DEMO"
    analysis_type: str                         # "vegetation" | "water"
    index_name: str                            # "NDVI" | "NDWI"
    bbox: tuple
    baseline_index: np.ndarray
    recent_index: np.ndarray
    change: np.ndarray
    anomalies: list[dict] = field(default_factory=list)
    bounds: tuple | None = None                # (south, west, north, east)
    baseline_scenes: list[dict] = field(default_factory=list)
    recent_scenes: list[dict] = field(default_factory=list)
    cir_baseline: np.ndarray | None = None
    cir_recent: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    water_summary: dict | None = None          # ESTIMATED open-water extent (NDWI > 0)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    aoi_geometry: dict | None = None           # user's polygon (EPSG:4326)
    aoi_mask: np.ndarray | None = None         # True = inside polygon, on analysis grid


def _scene_meta(s: dict) -> dict:
    return {"id": s["id"], "date": s["date"], "cloud": s["cloud"], "tile": s["tile"]}


def _cir(red, green, nir) -> np.ndarray:
    """Colour-infrared composite (NIR, Red, Green) scaled to uint8 for display."""
    rgb = np.dstack([nir, red, green])
    rgb = np.nan_to_num(rgb, nan=0.0)
    hi = np.nanpercentile(rgb[rgb > 0], 98) if (rgb > 0).any() else 1.0
    return (np.clip(rgb / max(hi, 1e-6), 0, 1) * 255).astype("uint8")


def _geo_helpers(transform, crs, shape):
    from rasterio.warp import transform as warp_transform

    def to_lonlat(col: float, row: float):
        x, y = transform * (col, row)
        lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
        return lon[0], lat[0]

    h, w = shape
    corners = [to_lonlat(0, 0), to_lonlat(w, 0), to_lonlat(w, h), to_lonlat(0, h)]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    bounds = (min(lats), min(lons), max(lats), max(lons))  # S, W, N, E
    return to_lonlat, bounds, abs(transform.a * transform.e)


def attach_water_extent(regions: list[dict], baseline: np.ndarray, recent: np.ndarray,
                        pixel_area_m2: float) -> None:
    """Add the estimated open-water extent change (NDWI > 0 masks) to each region."""
    for r in regions:
        r["water_extent"] = wat.region_water_extent(
            baseline, recent, r["bbox_px"], r["mask_px"], pixel_area_m2)


def run_change_analysis(bbox, analysis_type: str, baseline_range: tuple[date, date],
                        recent_range: tuple[date, date], max_cloud: float,
                        threshold: float, min_area_ha: float,
                        aoi_geometry: dict | None = None) -> AnalysisResult:
    """Real Sentinel-2 analysis. Raises sat.SatelliteError with friendly text.

    `bbox` only decides which window of each scene is read. When `aoi_geometry` (the
    user's GeoJSON polygon, EPSG:4326) is given it is rasterised onto the analysis grid
    and every pixel outside it is excluded from the composites, the change map and the
    anomaly detection."""
    sat.validate_request(bbox, [baseline_range, recent_range])
    kind = analysis_type
    index_name = "NDVI" if kind == "vegetation" else "NDWI"

    rec_all = sat.search_scenes(bbox, *recent_range, max_cloud)
    base_all = sat.search_scenes(bbox, *baseline_range, max_cloud)
    if not rec_all or not base_all:
        which = "recent" if not rec_all else "baseline"
        raise sat.SatelliteError(
            f"No usable Sentinel-2 images were found for the {which} period with cloud "
            f"cover below {max_cloud:.0f}%. Try a longer date range or allow more cloud cover.")
    tile = sat.pick_common_tile(rec_all, base_all)
    if tile is None:
        raise sat.SatelliteError(
            "The two periods have no Sentinel-2 tile in common over this area. "
            "Try wider date ranges or a smaller area that sits inside one tile.")
    rec_sel = [s for s in rec_all if s["tile"] == tile][:config.MAX_SCENES_PER_PERIOD]
    base_sel = [s for s in base_all if s["tile"] == tile][:config.MAX_SCENES_PER_PERIOD]

    notes: list[str] = []

    def build(selected):
        idx_stack, used, last = [], [], None
        for s in selected:
            try:
                d = sat.load_scene(s["item"], bbox)
            except sat.SatelliteError as exc:
                notes.append(str(exc))
                continue
            if last is not None and d["red"].shape != last["red"].shape:
                notes.append(f"Scene {s['id']} skipped (grid mismatch).")
                continue
            if kind == "vegetation":
                idx = veg.compute_ndvi(d["red"], d["nir"])
            else:
                idx = wat.compute_ndwi(d["green"], d["nir"])
            idx_stack.append(veg.masked_index(idx, d["scl"]))
            used.append(_scene_meta(s))
            last = d if last is None else last  # first (newest) scene = preview/grid
        return idx_stack, used, last

    base_stack, base_used, base_ref = build(base_sel)
    rec_stack, rec_used, rec_ref = build(rec_sel)
    if not base_stack or not rec_stack:
        raise sat.SatelliteError(
            "Images were found but none could be read for this area. Try other dates.")
    if (base_ref["red"].shape != rec_ref["red"].shape
            or base_ref["transform"] != rec_ref["transform"]):
        raise sat.SatelliteError("The two periods use different pixel grids; try another area.")

    # Polygon mask on the analysis grid, using the raster's own CRS and transform.
    shape = rec_ref["red"].shape
    if aoi_geometry is not None:
        aoi_mask = sat.build_aoi_mask(aoi_geometry, rec_ref["transform"], rec_ref["crs"], shape)
        if int(aoi_mask.sum()) == 0:
            raise sat.SatelliteError(
                "The selected polygon is too small to cover any 10 m pixel. Draw a larger area.")
    else:
        aoi_mask = np.ones(shape, dtype=bool)
    for arr in base_stack + rec_stack:
        arr[~aoi_mask] = np.nan                      # exclude pixels outside the polygon

    base_c, rec_c = veg.composite(base_stack), veg.composite(rec_stack)
    for name, comp in (("baseline", base_c), ("recent", rec_c)):
        vf = veg.valid_fraction(comp, aoi_mask)
        if vf < config.MIN_VALID_FRACTION:
            raise sat.SatelliteError(
                f"Too few clear pixels in the {name} period ({vf:.0%} usable after cloud "
                "masking). Try a longer date range or a higher cloud-cover limit.")
    if len(base_stack) < 2 or len(rec_stack) < 2:
        notes.append("Only one usable image in at least one period: results are more "
                     "sensitive to single-date noise.")

    change = veg.compute_change(rec_c, base_c)
    to_lonlat, bounds, px_area = _geo_helpers(rec_ref["transform"], rec_ref["crs"],
                                              rec_ref["red"].shape)
    kw = dict(pixel_area_m2=px_area, to_lonlat=to_lonlat, min_area_ha=min_area_ha,
              aoi_mask=aoi_mask)
    regions = detect_regions(change, threshold, "decrease", **kw)
    water_sum = None
    if kind == "water":
        regions += detect_regions(change, threshold, "increase", **kw)
        water_sum = wat.water_extent_change(base_c, rec_c, px_area)
        attach_water_extent(regions, base_c, rec_c, px_area)

    res = AnalysisResult(
        mode="REAL", analysis_type=kind, index_name=index_name, bbox=tuple(bbox),
        baseline_index=base_c, recent_index=rec_c, change=change, anomalies=regions,
        bounds=bounds, baseline_scenes=base_used, recent_scenes=rec_used,
        cir_baseline=_cir(base_ref["red"], base_ref["green"], base_ref["nir"]),
        cir_recent=_cir(rec_ref["red"], rec_ref["green"], rec_ref["nir"]),
        params={"threshold": threshold, "min_area_ha": min_area_ha,
                "max_cloud": max_cloud, "tile": tile,
                "baseline_range": baseline_range, "recent_range": recent_range},
        water_summary=water_sum, notes=notes,
        source="Sentinel-2 L2A via Microsoft Planetary Computer",
        aoi_geometry=aoi_geometry, aoi_mask=aoi_mask)
    if aoi_geometry is not None:
        res.notes.append(f"Polygon mask applied: {aoi_mask.mean():.0%} of the bounding "
                         "rectangle lies inside your polygon; the rest was excluded.")
    describe_regions(res)
    return res


def run_demo_analysis(bbox, analysis_type: str, threshold: float,
                      min_area_ha: float, aoi_geometry: dict | None = None) -> AnalysisResult:
    """DEMO MODE with synthetic arrays. Every output is labelled DEMO."""
    d = sat.demo_scenario(bbox, analysis_type)
    w, s, e, n = bbox
    size = d["baseline"].shape[0]
    px_w, px_h = (e - w) / size, (n - s) / size
    import math
    px_area = (px_w * 111_320 * math.cos(math.radians((s + n) / 2))) * (px_h * 110_540)

    def to_lonlat(col, row):
        return w + col * px_w, n - row * px_h

    if aoi_geometry is not None:   # demo grid is a plain lon/lat grid: no CRS projection
        ring_px = [((x - w) / px_w, (n - y) / px_h) for x, y in aoi_geometry["coordinates"][0]]
        aoi_mask = sat.polygon_mask_from_pixel_ring(ring_px, (size, size))
    else:
        aoi_mask = np.ones((size, size), dtype=bool)
    baseline = np.where(aoi_mask, d["baseline"], np.nan).astype("float32")
    recent = np.where(aoi_mask, d["recent"], np.nan).astype("float32")

    change = recent - baseline
    kw = dict(pixel_area_m2=px_area, to_lonlat=to_lonlat, min_area_ha=min_area_ha,
              aoi_mask=aoi_mask)
    regions = detect_regions(change, threshold, "decrease", **kw)
    water_sum = None
    if analysis_type == "water":
        regions += detect_regions(change, threshold, "increase", **kw)
        water_sum = wat.water_extent_change(baseline, recent, px_area)
        attach_water_extent(regions, baseline, recent, px_area)
    res = AnalysisResult(
        mode="DEMO", analysis_type=analysis_type,
        index_name="NDVI" if analysis_type == "vegetation" else "NDWI",
        bbox=tuple(bbox), baseline_index=baseline, recent_index=recent,
        change=change, anomalies=regions, bounds=(s, w, n, e),
        params={"threshold": threshold, "min_area_ha": min_area_ha},
        water_summary=water_sum,
        notes=["DEMO MODE: synthetic data, not a satellite observation."],
        source="DEMO — synthetic data (not a satellite observation)",
        aoi_geometry=aoi_geometry, aoi_mask=aoi_mask)
    describe_regions(res)
    return res


def describe_regions(res: AnalysisResult) -> None:
    """Attach type, reason text and index-specific fields to every region."""
    idx = res.index_name
    for r in res.anomalies:
        verb = "decreased" if r["direction"] == "decrease" else "increased"
        head = (f"Mean {idx} {verb} by {abs(r['mean_change']):.2f} "
                f"(threshold {abs(res.params.get('threshold', 0)):.2f}) over "
                f"{r['area_ha']:.2f} ha between the baseline and recent periods.")
        if res.analysis_type == "vegetation":
            r["type"] = "Vegetation decrease"
            r["reason"] = (head + " Seasonal change, harvest, tillage, drought or residual "
                           "cloud can produce similar signals; the cause is not identified.")
            continue
        # Water: report the NDWI INDEX change and, separately, the ESTIMATED open-water
        # extent change (NDWI > 0 masks). One does not imply the other.
        r["type"] = f"NDWI {r['direction']} (index change)"
        we = r.get("water_extent")
        if we:
            extent = (f" Separately, estimated open-water extent in this region (NDWI > 0 "
                      f"mask) went from {we['baseline_water_ha']:.2f} ha to "
                      f"{we['recent_water_ha']:.2f} ha (lost {we['lost_ha']:.2f} ha, gained "
                      f"{we['gained_ha']:.2f} ha).")
            if we["lost_ha"] == 0 and we["gained_ha"] == 0:
                extent += " The NDWI > 0 water mask did not change here."
        else:
            extent = ""
        r["reason"] = (head + " This is a change in a reflectance index, not a confirmed "
                       "change in water extent or water quality." + extent +
                       " Water-level change, wetness, turbidity, vegetation, sun-glint or "
                       "residual cloud can produce similar NDWI changes; the cause is not "
                       "identified.")
