"""Folium map builders and raster-to-overlay helpers."""
from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

import config

ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                "MapServer/tile/{z}/{y}/{x}")
DEFAULT_CENTER = (50.0, 30.5)

# ------------------------------------------------------------ colour helpers
_NDVI_STOPS = [(-0.2, (140, 100, 60)), (0.1, (235, 220, 150)),
               (0.4, (150, 200, 90)), (0.8, (20, 100, 50))]
_NDWI_STOPS = [(-0.6, (200, 180, 130)), (0.0, (235, 235, 235)),
               (0.3, (110, 170, 220)), (0.8, (20, 70, 160))]
_CHANGE_STOPS = [(-0.5, (179, 38, 30)), (-0.2, (230, 150, 60)),
                 (0.0, (245, 245, 240)), (0.2, (150, 200, 120)), (0.5, (31, 77, 54))]


def colorize(values: np.ndarray, stops: list[tuple[float, tuple]],
             alpha: int = 200) -> np.ndarray:
    """Map float array to RGBA uint8; NaN becomes fully transparent."""
    xs = np.array([s[0] for s in stops])
    rgb = np.stack([np.interp(np.nan_to_num(values, nan=xs[0]), xs,
                              [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    a = np.where(np.isfinite(values), alpha, 0).astype("uint8")
    return np.dstack([rgb.astype("uint8"), a])


def index_rgba(index: np.ndarray, name: str) -> np.ndarray:
    return colorize(index, _NDVI_STOPS if name == "NDVI" else _NDWI_STOPS)


def change_rgba(change: np.ndarray) -> np.ndarray:
    return colorize(change, _CHANGE_STOPS)


def to_data_uri(arr: np.ndarray) -> str:
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def legend_html(index_name: str) -> str:
    return (
        "<div style='font-size:0.85rem;line-height:1.7'>"
        "<b>Change map (recent − baseline)</b><br>"
        "<span style='color:#B3261E'>■</span> Strong decrease &nbsp;"
        "<span style='color:#E69640'>■</span> Moderate decrease &nbsp;"
        "<span style='color:#BBBBB5'>■</span> No notable change &nbsp;"
        "<span style='color:#96C878'>■</span> Increase<br>"
        "<b>Anomaly outlines</b><br>"
        f"<span style='color:{config.COLORS['Normal']}'>▭</span> Normal change &nbsp;"
        f"<span style='color:{config.COLORS['Attention']}'>▭</span> Requires attention &nbsp;"
        f"<span style='color:{config.COLORS['High']}'>▭</span> High priority<br>"
        f"<i>{index_name} changes show reflectance differences, not their cause.</i></div>")


# ------------------------------------------------------------- map builders
def _base_map(center=DEFAULT_CENTER, zoom=6):
    import folium
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None,
                   control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer(ESRI_IMAGERY, attr="Esri World Imagery", name="Satellite basemap").add_to(m)
    return m


def draw_map(center=DEFAULT_CENTER, zoom=6, areas: list[dict] | None = None,
             highlight_geom: dict | None = None):
    """Map with polygon/rectangle drawing tools (streamlit-folium reads the output)."""
    import folium
    from folium.plugins import Draw
    m = _base_map(center, zoom)
    for a in areas or []:
        folium.GeoJson(json.loads(a["geometry"]), name=a["name"],
                       style_function=lambda _f: {"color": config.COLORS["secondary"],
                                                  "weight": 2, "fillOpacity": 0.08},
                       tooltip=a["name"]).add_to(m)
    if highlight_geom:
        folium.GeoJson(highlight_geom, style_function=lambda _f: {
            "color": config.COLORS["primary"], "weight": 3, "fillOpacity": 0.1}).add_to(m)
    Draw(export=False, draw_options={"polygon": True, "rectangle": True,
                                     "polyline": False, "circle": False,
                                     "marker": False, "circlemarker": False},
         edit_options={"edit": False}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def extract_geometry(map_output: dict | None) -> dict | None:
    """Pull the most recent drawn polygon (GeoJSON geometry) from st_folium output."""
    if not map_output:
        return None
    drawings = map_output.get("all_drawings") or []
    cand = drawings[-1] if drawings else map_output.get("last_active_drawing")
    geom = (cand or {}).get("geometry")
    if geom and geom.get("type") == "Polygon":
        return geom
    return None


def geometry_bbox(geom: dict) -> tuple[float, float, float, float]:
    ring = geom["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def geometry_center(geom: dict) -> tuple[float, float]:
    w, s, e, n = geometry_bbox(geom)
    return (s + n) / 2, (w + e) / 2


def overview_map(areas: list[dict], anomalies: list[dict]):
    import folium
    m = _base_map(zoom=6)
    pts = []
    for a in areas:
        g = json.loads(a["geometry"])
        label = a["name"] + (" (DEMO)" if a["is_demo"] else "")
        folium.GeoJson(g, tooltip=label, style_function=lambda _f: {
            "color": config.COLORS["primary"], "weight": 2, "fillOpacity": 0.12}).add_to(m)
        w, s, e, n = geometry_bbox(g)
        pts += [(s, w), (n, e)]
    for an in anomalies:
        demo = bool(an["is_demo"])
        folium.CircleMarker(
            [an["latitude"], an["longitude"]], radius=8,
            color=config.COLORS[an["priority"]], fill=True, fill_opacity=0.8,
            dash_array="4" if demo else None,
            tooltip=f"{an['anomaly_code']} · {config.PRIORITY_LABELS[an['priority']]}"
                    + (" · DEMO" if demo else "")).add_to(m)
        pts.append((an["latitude"], an["longitude"]))
    if pts:
        m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                      [max(p[0] for p in pts), max(p[1] for p in pts)]])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def result_map(result, selected_idx: int | None = None):
    """Analysis result: baseline/recent colour-infrared, index map, change map,
    anomaly outlines (coloured by priority tier)."""
    import folium
    s, w, n, e = result.bounds
    bounds = [[s, w], [n, e]]
    m = _base_map(((s + n) / 2, (w + e) / 2), 13)
    demo = result.mode == "DEMO"
    sfx = " (DEMO)" if demo else ""
    layers = []
    if result.cir_baseline is not None:
        layers.append(("Baseline image (colour-infrared)", to_data_uri(result.cir_baseline), 0.9, False))
    if result.cir_recent is not None:
        layers.append(("Recent image (colour-infrared)", to_data_uri(result.cir_recent), 0.9, False))
    layers.append((f"Baseline {result.index_name}{sfx}",
                   to_data_uri(index_rgba(result.baseline_index, result.index_name)), 1, False))
    layers.append((f"Recent {result.index_name}{sfx}",
                   to_data_uri(index_rgba(result.recent_index, result.index_name)), 1, False))
    layers.append((f"Change detection{sfx}", to_data_uri(change_rgba(result.change)), 1, True))
    for name, uri, op, show in layers:
        folium.raster_layers.ImageOverlay(image=uri, bounds=bounds, name=name,
                                          opacity=op, show=show).add_to(m)
    if getattr(result, "aoi_geometry", None):
        folium.GeoJson(result.aoi_geometry, name="Selected area (polygon mask)",
                       style_function=lambda _f: {"color": config.COLORS["primary"],
                                                  "weight": 3, "fill": False,
                                                  "dashArray": "6"}).add_to(m)
    for i, a in enumerate(result.anomalies):
        col = config.COLORS[a["priority"]]
        folium.GeoJson(
            a["polygon"], name=f"Anomaly {i + 1}",
            style_function=lambda _f, c=col, sel=(i == selected_idx): {
                "color": c, "weight": 4 if sel else 2, "fillOpacity": 0.05},
            tooltip=f"Region {i + 1} · {config.PRIORITY_LABELS[a['priority']]} · "
                    f"{a['area_ha']:.1f} ha").add_to(m)
    m.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def anomaly_map(anomalies: list[dict], center: tuple[float, float] | None = None, zoom=12):
    """Saved anomalies; marker tooltip = anomaly code (used for click selection)."""
    import folium
    if center is None and anomalies:
        center = (anomalies[0]["latitude"], anomalies[0]["longitude"])
    m = _base_map(center or DEFAULT_CENTER, zoom)
    for an in anomalies:
        if an.get("polygon"):
            folium.GeoJson(json.loads(an["polygon"]), style_function=lambda _f, c=config.COLORS[an["priority"]]: {
                "color": c, "weight": 2, "fillOpacity": 0.1}).add_to(m)
        folium.Marker([an["latitude"], an["longitude"]], tooltip=an["anomaly_code"],
                      icon=folium.Icon(color="red" if an["priority"] == "High"
                                       else "orange" if an["priority"] == "Attention"
                                       else "green")).add_to(m)
    return m


def point_map(lat: float, lon: float, label: str, zoom: int = 15):
    import folium
    m = _base_map((lat, lon), zoom)
    folium.Marker([lat, lon], tooltip=label).add_to(m)
    return m
