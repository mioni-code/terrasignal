"""Business-requested monitoring areas: geometry helpers and CRUD wrappers."""
from __future__ import annotations

import json
import math

import config
import database as db


def polygon_area_ha(geom: dict) -> float:
    """Planar-approximation area of a lon/lat polygon in hectares."""
    return db._approx_ha(geom)


def create_area(name: str, geometry: dict | None, category: str, description: str,
                frequency: str, budget: float | None) -> int:
    """Validate and save a new monitoring area. Raises ValueError (friendly text)."""
    db.validate_area(name, category, frequency, budget, geometry)
    if geometry is None:
        raise ValueError("Please draw a polygon or rectangle on the map.")
    ha = polygon_area_ha(geometry)
    if ha <= 0:
        raise ValueError("The drawn shape has no area. Please draw it again.")
    return db.add_area(name, geometry, category, description, frequency, budget, ha)


def area_geometry(area: dict) -> dict:
    return json.loads(area["geometry"])


def area_bbox(area: dict) -> tuple[float, float, float, float]:
    ring = area_geometry(area)["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def area_centroid(area: dict) -> tuple[float, float]:
    w, s, e, n = area_bbox(area)
    return (s + n) / 2, (w + e) / 2


def frequency_note() -> str:
    return config.FREQUENCY_NOTE
