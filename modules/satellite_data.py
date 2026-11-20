"""Sentinel-2 L2A access through Microsoft Planetary Computer (STAC).

Only the requested AOI window is read from the cloud-optimised GeoTIFFs; entire
scenes are never downloaded.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

import config

BBox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)


class SatelliteError(Exception):
    """User-presentable satellite access problem."""


def libs_available() -> tuple[bool, str]:
    missing = []
    for mod in ("pystac_client", "planetary_computer", "rasterio"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return False, "Missing libraries: " + ", ".join(missing)
    return True, "ok"


def check_connection(timeout: int = 8) -> tuple[bool, str]:
    """Light connectivity check against the STAC root (no data downloaded)."""
    ok, msg = libs_available()
    if not ok:
        return False, msg
    try:
        import requests
        r = requests.get(config.STAC_URL, timeout=timeout)
        r.raise_for_status()
        return True, "Planetary Computer reachable"
    except Exception as exc:
        return False, f"Planetary Computer unreachable ({type(exc).__name__})"


# --------------------------------------------------------------- validation
def bbox_area_km2(bbox: BBox) -> float:
    w, s, e, n = bbox
    lat = math.radians((s + n) / 2)
    return abs(e - w) * 111.32 * math.cos(lat) * abs(n - s) * 110.54


def validate_request(bbox: BBox | None, ranges: list[tuple[date, date]]) -> None:
    if bbox is None:
        raise SatelliteError("Please draw an Area of Interest on the map first.")
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise SatelliteError("The selected area has invalid coordinates.")
    km2 = bbox_area_km2(bbox)
    if km2 > config.MAX_AOI_KM2:
        raise SatelliteError(
            f"The selected area is about {km2:,.0f} km², above the {config.MAX_AOI_KM2:.0f}"
            " km² limit for this prototype. Please draw a smaller area.")
    for start, end in ranges:
        if start > end:
            raise SatelliteError("A date range starts after it ends. Please fix the dates.")
        if end > date.today():
            raise SatelliteError("A date range ends in the future. Please choose past dates.")


# ------------------------------------------------------------------- search
def search_scenes(bbox: BBox, start: date, end: date, max_cloud: float,
                  limit: int = config.MAX_SEARCH_ITEMS) -> list[dict[str, Any]]:
    """Return signed STAC items (newest first) as light dicts."""
    ok, msg = libs_available()
    if not ok:
        raise SatelliteError(msg)
    try:
        import planetary_computer as pc
        from pystac_client import Client

        catalog = Client.open(config.STAC_URL, modifier=pc.sign_inplace)
        search = catalog.search(
            collections=[config.COLLECTION], bbox=list(bbox),
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": float(max_cloud)}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=limit)
        items = list(search.items())
    except Exception as exc:
        raise SatelliteError(
            f"Could not search Planetary Computer ({type(exc).__name__}: {exc}).") from exc

    scenes = []
    for it in items:
        props = it.properties
        tile = props.get("s2:mgrs_tile") or (it.id.split("_")[1] if "_" in it.id else it.id)
        scenes.append({
            "id": it.id,
            "date": it.datetime.date(),
            "cloud": float(props.get("eo:cloud_cover", float("nan"))),
            "tile": tile,
            "item": it,
        })
    return scenes


# ------------------------------------------------------------ windowed read
def _read_window(href: str, bbox: BBox, out_shape: tuple[int, int] | None,
                 nearest: bool = False):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    with rasterio.open(href) as src:
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = src.window(l, b, r, t)
        col0, row0 = math.floor(win.col_off), math.floor(win.row_off)
        w, h = math.ceil(win.width), math.ceil(win.height)
        win = Window(col0, row0, w, h).intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            raise SatelliteError("The AOI does not overlap this scene.")
        res = Resampling.nearest if nearest else Resampling.bilinear
        data = src.read(1, window=win, out_shape=out_shape, resampling=res)
        tr = src.window_transform(win)
        if out_shape is not None:
            tr = tr * Affine.scale(win.width / out_shape[1], win.height / out_shape[0])
        return data, tr, src.crs


def _reflectance(item, key: str, dn: np.ndarray) -> np.ndarray:
    """Digital numbers -> surface reflectance, honouring the STAC offset."""
    scale, offset = 1e-4, None
    rb = item.assets[key].extra_fields.get("raster:bands")
    if rb:
        scale = rb[0].get("scale", scale)
        offset = rb[0].get("offset")
    if offset is None:
        pb = str(item.properties.get("s2:processing_baseline", "00.00"))
        offset = -1000.0 if pb >= "04.00" else 0.0
    out = (dn.astype("float32") * scale) + float(offset) * scale
    out[dn == 0] = np.nan  # nodata
    return np.clip(out, 0, 1.5)


def load_scene(item, bbox: BBox) -> dict[str, Any]:
    """Read B04, B08, B03 and SCL for the AOI window of one scene."""
    needed = ["B04", "B08", "B03", "SCL"]
    missing = [k for k in needed if k not in item.assets]
    if missing:
        raise SatelliteError(f"Scene {item.id} is missing bands: {', '.join(missing)}")
    try:
        red_dn, transform, crs = _read_window(item.assets["B04"].href, bbox, None)
        shape = red_dn.shape
        nir_dn, _, _ = _read_window(item.assets["B08"].href, bbox, shape)
        grn_dn, _, _ = _read_window(item.assets["B03"].href, bbox, shape)
        scl, _, _ = _read_window(item.assets["SCL"].href, bbox, shape, nearest=True)
    except SatelliteError:
        raise
    except Exception as exc:
        raise SatelliteError(f"Could not read scene {item.id}: {exc}") from exc
    return {
        "red": _reflectance(item, "B04", red_dn),
        "nir": _reflectance(item, "B08", nir_dn),
        "green": _reflectance(item, "B03", grn_dn),
        "scl": scl,
        "transform": transform,
        "crs": crs,
    }


def pick_common_tile(recent: list[dict], baseline: list[dict]) -> str | None:
    """Tile with scenes in both periods (so both share one pixel grid)."""
    def counts(sc):
        d: dict[str, int] = {}
        for s in sc:
            d[s["tile"]] = d.get(s["tile"], 0) + 1
        return d
    cr, cb = counts(recent), counts(baseline)
    common = set(cr) & set(cb)
    if not common:
        return None
    return max(common, key=lambda t: (min(cr[t], cb[t]), cr[t] + cb[t]))


# --------------------------------------------------------- AOI polygon masking
def polygon_mask_from_pixel_ring(ring_px, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask (True = inside) of a polygon given as (col, row) pixel-space
    vertices. Even-odd rule evaluated at pixel centres, matching rasterio's default
    (all_touched=False) behaviour. Pure numpy so it is testable without rasterio."""
    pts = [(float(x), float(y)) for x, y in ring_px]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise SatelliteError("The selected polygon needs at least three points.")
    h, w = shape
    py, px = np.mgrid[0:h, 0:w]
    px = px + 0.5
    py = py + 0.5
    inside = np.zeros(shape, dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yi != yj:
            crosses = (yi > py) != (yj > py)
            x_int = (xj - xi) * (py - yi) / (yj - yi) + xi
            inside ^= crosses & (px < x_int)
        j = i
    return inside


def _default_project(crs):
    """Return f(lons, lats) -> (xs, ys) projecting EPSG:4326 into the raster CRS."""
    from rasterio.warp import transform as warp_transform

    def project(lons, lats):
        return warp_transform("EPSG:4326", crs, list(lons), list(lats))
    return project


def build_aoi_mask(geometry: dict, transform, crs, shape: tuple[int, int],
                   project=None) -> np.ndarray:
    """Rasterise the user's GeoJSON polygon (EPSG:4326) onto the analysis grid.

    1. Vertices are projected from lon/lat into the raster's own CRS.
    2. They are converted to pixel space with the inverse of the raster transform.
    3. Pixels whose centres fall inside the polygon are True.
    `project` can be injected (tests); by default rasterio.warp is used.
    """
    if not geometry or geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
        raise SatelliteError("The area of interest must be a polygon.")
    ring = geometry["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    xs, ys = (project or _default_project(crs))(lons, lats)
    inv = ~transform
    ring_px = [inv * (x, y) for x, y in zip(xs, ys)]
    return polygon_mask_from_pixel_ring(ring_px, shape)


# ------------------------------------------------------------- DEMO scenario
def demo_scenario(bbox: BBox, kind: str = "vegetation", size: int = 120,
                  seed: int = 7) -> dict[str, Any]:
    """Synthetic arrays for DEMO MODE. NOT satellite data; always labelled DEMO."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 0.62 + 0.08 * np.sin(xx / 14.0) + rng.normal(0, 0.02, (size, size))
    recent = base + rng.normal(0, 0.02, (size, size))
    for cy, cx, r, drop in [(40, 45, 10, 0.45), (85, 80, 6, 0.30), (20, 95, 3, 0.35)]:
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        recent[m] -= drop
    if kind == "water":
        base = np.where((yy - 60) ** 2 + (xx - 60) ** 2 < 35 ** 2, 0.55, -0.35) \
            + rng.normal(0, 0.02, (size, size))
        recent = base.copy()
        recent[(yy - 60) ** 2 + (xx - 60) ** 2 > 28 ** 2] = -0.35
    return {"baseline": base.astype("float32"), "recent": recent.astype("float32"),
            "bbox": bbox}
