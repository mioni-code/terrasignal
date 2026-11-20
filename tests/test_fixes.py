"""Tests for: polygon masking, draw_map arguments, NDWI vs water-extent terminology and
report wording. Offline; run:  python tests/test_fixes.py  (or pytest)."""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
import sys
import tempfile
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "fixes.db"
import database as db  # noqa: E402
from modules import (anomaly_detection as ad, gemini_service as gem,  # noqa: E402
                     map_interface, report_generator as rg, satellite_data as sat,
                     water_analysis as wat)


# ------------------------------------------------------------------ helpers
class FakeAffine:
    """Minimal stand-in for affine.Affine (rasterio is not installed in CI here)."""

    def __init__(self, a, c, e, f):
        self.a, self.c, self.e, self.f = a, c, e, f

    def __mul__(self, xy):
        x, y = xy
        return (self.a * x + self.c, self.e * y + self.f)

    def __invert__(self):
        return FakeAffine(1 / self.a, -self.c / self.a, 1 / self.e, -self.f / self.e)

    def __eq__(self, other):
        return (self.a, self.c, self.e, self.f) == (other.a, other.c, other.e, other.f)

    __hash__ = None


# 100x100 grid, 10 m pixels, north-up, in a fake projected CRS where
# x = (lon-30)*1e5 and y = (lat-50)*1e5  (so the grid spans lon 30..30.01, lat 50..50.01)
TRANSFORM = FakeAffine(10.0, 0.0, -10.0, 1000.0)
SHAPE = (100, 100)


def fake_project(lons, lats):
    return [(lo - 30.0) * 1e5 for lo in lons], [(la - 50.0) * 1e5 for la in lats]


def fake_to_lonlat(col, row):
    x, y = TRANSFORM * (col, row)
    return 30.0 + x / 1e5, 50.0 + y / 1e5


# lower-left triangle of the grid (array pixels with row > col)
TRIANGLE = {"type": "Polygon", "coordinates": [[[30.0, 50.0], [30.01, 50.0],
                                                [30.0, 50.01], [30.0, 50.0]]]}
BBOX = (30.0, 50.0, 30.01, 50.01)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# ------------------------------------------------------- 1. polygon masking
def test_polygon_mask_excludes_outside_pixels():
    square = [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]
    m = sat.polygon_mask_from_pixel_ring(square, (50, 50))
    assert m.sum() == 400                          # 20 x 20 pixel centres
    assert m[15, 15] and m[10, 10] and m[29, 29]
    assert not m[5, 5] and not m[40, 40] and not m[9, 15] and not m[30, 15]
    # concave polygon: the notch must stay outside
    ell = [(0, 0), (40, 0), (40, 10), (10, 10), (10, 40), (0, 40), (0, 0)]
    m2 = sat.polygon_mask_from_pixel_ring(ell, (50, 50))
    assert m2[5, 5] and m2[5, 35] and m2[35, 5] and not m2[30, 30]
    raises(sat.SatelliteError, sat.polygon_mask_from_pixel_ring, [(0, 0), (1, 1)], (5, 5))


def test_polygon_mask_uses_crs_and_transform():
    m = sat.build_aoi_mask(TRIANGLE, TRANSFORM, "FAKE:CRS", SHAPE, project=fake_project)
    # lower-left triangle in map space -> rows > cols in a north-up array
    assert m[90, 5] and m[60, 20] and not m[5, 90] and not m[20, 60]
    assert not m[0, 99] and m[99, 0]                       # top-right out, bottom-left in
    assert 0.45 < m.mean() < 0.55
    # Using lon/lat directly with the projected transform (i.e. ignoring the CRS) would
    # produce a near-empty mask, so a large mask proves the projection step ran.
    wrong = sat.build_aoi_mask(TRIANGLE, TRANSFORM, "FAKE:CRS", SHAPE,
                               project=lambda lo, la: (lo, la))
    assert wrong.sum() < 5
    # polygon far away from the grid -> nothing inside
    far = {"type": "Polygon", "coordinates": [[[31, 51], [31.01, 51], [31, 51.01], [31, 51]]]}
    assert sat.build_aoi_mask(far, TRANSFORM, "FAKE:CRS", SHAPE, project=fake_project).sum() == 0
    raises(sat.SatelliteError, sat.build_aoi_mask, {"type": "Point"}, TRANSFORM, "x", SHAPE)


def test_detect_regions_respects_aoi_mask():
    ch = np.zeros(SHAPE, dtype="float32")
    ch[60:80, 10:30] = -0.5       # inside the triangle
    ch[10:30, 60:80] = -0.5       # inside the bounding box but outside the triangle
    mask = sat.build_aoi_mask(TRIANGLE, TRANSFORM, "FAKE", SHAPE, project=fake_project)
    kw = dict(pixel_area_m2=100.0, to_lonlat=fake_to_lonlat, min_area_ha=0.5)
    assert len(ad.detect_regions(ch, 0.2, "decrease", **kw)) == 2
    masked = ad.detect_regions(ch, 0.2, "decrease", aoi_mask=mask, **kw)
    assert len(masked) == 1 and masked[0]["longitude"] < 30.0045
    raises(ValueError, ad.detect_regions, ch, 0.2, "decrease", aoi_mask=np.ones((3, 3), bool), **kw)


def _fake_pipeline(monkey_kind: str, recent_patches: list):
    """Patch the network/raster edges so the REAL run_change_analysis can run offline."""
    h, w = SHAPE

    def scene(period):
        red = np.full(SHAPE, 0.1, "float32")
        nir = np.full(SHAPE, 0.5, "float32")
        grn = np.full(SHAPE, 0.3, "float32")
        if monkey_kind == "water":
            nir = np.full(SHAPE, 0.1, "float32")
        if period == "recent":
            for (r0, r1, c0, c1, g, n_) in recent_patches:
                grn[r0:r1, c0:c1] = g
                nir[r0:r1, c0:c1] = n_
        return {"red": red, "nir": nir, "green": grn, "scl": np.full(SHAPE, 4, "uint8"),
                "transform": FakeAffine(10.0, 0.0, -10.0, 1000.0), "crs": "FAKE:CRS"}

    def fake_search(bbox, start, end, max_cloud, limit=0):
        period = "recent" if start.year >= 2026 else "baseline"
        return [{"id": f"{period}-{i}", "date": date(start.year, start.month, 5 + i),
                 "cloud": 5.0, "tile": "T36UUA", "item": {"period": period}} for i in range(2)]

    saved = (sat.search_scenes, sat.load_scene, sat._default_project, ad._geo_helpers)
    sat.search_scenes = fake_search
    sat.load_scene = lambda item, bbox: scene(item["period"])
    sat._default_project = lambda crs: fake_project
    ad._geo_helpers = lambda tr, crs, shape: (
        fake_to_lonlat, (50.0, 30.0, 50.01, 30.01), 100.0)

    def restore():
        sat.search_scenes, sat.load_scene, sat._default_project, ad._geo_helpers = saved
    return restore


RANGES = dict(baseline_range=(date(2025, 10, 1), date(2025, 10, 31)),
              recent_range=(date(2026, 10, 1), date(2026, 11, 10)))


def test_polygon_aoi_does_not_analyse_whole_bounding_rectangle():
    # NDVI drops in one block inside the triangle and one outside it (still in the bbox)
    patches = [(60, 80, 10, 30, 0.3, 0.2), (10, 30, 60, 80, 0.3, 0.2)]
    restore = _fake_pipeline("vegetation", patches)
    try:
        full = ad.run_change_analysis(BBOX, "vegetation", max_cloud=30, threshold=0.2,
                                      min_area_ha=0.5, **RANGES)
        poly = ad.run_change_analysis(BBOX, "vegetation", max_cloud=30, threshold=0.2,
                                      min_area_ha=0.5, aoi_geometry=TRIANGLE, **RANGES)
    finally:
        restore()
    assert len(full.anomalies) == 2                      # old behaviour: whole rectangle
    assert len(poly.anomalies) == 1                      # polygon: outside block ignored
    assert poly.aoi_mask is not None and 0.45 < poly.aoi_mask.mean() < 0.55
    # nothing outside the polygon survives anywhere in the outputs
    out = ~poly.aoi_mask
    for arr in (poly.baseline_index, poly.recent_index, poly.change):
        assert np.isnan(arr[out]).all()
        assert np.isfinite(arr[poly.aoi_mask]).all()
    a = poly.anomalies[0]
    assert a["longitude"] < 30.0045 and a["latitude"] < 50.0045   # lower-left half
    assert not poly.anomalies[0]["mask_px"].shape == ()           # footprint kept
    r0, r1, c0, c1 = a["bbox_px"]
    assert (poly.aoi_mask[r0:r1, c0:c1] | ~a["mask_px"]).all()    # footprint inside AOI
    assert any("Polygon mask applied" in n for n in poly.notes)
    # a polygon that misses the raster entirely gives a friendly error, not a crash
    far = {"type": "Polygon", "coordinates": [[[31, 51], [31.01, 51], [31, 51.01], [31, 51]]]}
    restore = _fake_pipeline("vegetation", patches)
    try:
        raises(sat.SatelliteError, ad.run_change_analysis, BBOX, "vegetation",
               max_cloud=30, threshold=0.2, min_area_ha=0.5, aoi_geometry=far, **RANGES)
    finally:
        restore()


def test_demo_polygon_is_masked_too():
    bbox = (30.40, 50.40, 30.45, 50.43)
    w, s, e, n = bbox
    tri = {"type": "Polygon", "coordinates": [[[w, n], [w, s], [e, s], [w, n]]]}
    full = ad.run_demo_analysis(bbox, "vegetation", 0.2, 0.05)
    poly = ad.run_demo_analysis(bbox, "vegetation", 0.2, 0.05, aoi_geometry=tri)
    assert len(full.anomalies) == 3 and len(poly.anomalies) == 2
    assert 0.4 < poly.aoi_mask.mean() < 0.6 and poly.mode == "DEMO"
    assert np.isnan(poly.change[~poly.aoi_mask]).all()


# ------------------------------------------------------ 2. map arguments
def test_map_function_arguments():
    tree = ast.parse((ROOT / "app.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "mi"]
    assert calls, "no mi.* calls found"
    for c in calls:
        fn = getattr(map_interface, c.func.attr)          # function must exist
        assert not any(isinstance(a, ast.Starred) for a in c.args), (
            f"mi.{c.func.attr} called with *args at line {c.lineno}")
        inspect.signature(fn).bind(*[None] * len(c.args),
                                   **{k.arg: None for k in c.keywords if k.arg})
    # the regression: saved-area map must pass center and zoom by keyword
    area_calls = [c for c in calls if c.func.attr == "draw_map"
                  and {"center", "zoom", "highlight_geom"} <= {k.arg for k in c.keywords}]
    assert area_calls, "saved-area draw_map call must use keyword center=, zoom="
    assert all(not c.args for c in area_calls)
    # and the old buggy call shape really would have failed
    raises(TypeError, inspect.signature(map_interface.draw_map).bind, 50.0, 30.0, zoom=13)
    inspect.signature(map_interface.draw_map).bind(center=(50.0, 30.0), zoom=13,
                                                   highlight_geom={})
    try:  # real call only if folium is installed
        import folium  # noqa: F401
        map_interface.draw_map(center=(50.0, 30.0), zoom=13, highlight_geom=TRIANGLE)
    except ImportError:
        pass


# ------------------------------------------- 3. NDWI index vs water extent
def test_ndwi_index_change_is_not_water_extent_change():
    b = np.full((10, 10), 0.5, "float32")
    r_still_water = np.full((10, 10), 0.2, "float32")      # NDWI fell 0.3 but still > 0
    r_dried = np.full((10, 10), -0.1, "float32")           # crossed below 0
    ha100 = 100.0                                           # 100 px * 1 ha each = 100 ha
    keep = wat.water_extent_change(b, r_still_water, 10_000.0)
    assert keep["lost_ha"] == 0 and keep["recent_water_ha"] == keep["baseline_water_ha"] == 100.0
    lost = wat.water_extent_change(b, r_dried, 10_000.0)
    assert lost["lost_ha"] == ha100 and lost["recent_water_ha"] == 0
    # region-level helper
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    reg = wat.region_water_extent(b, r_dried, (3, 7, 3, 7), mask, 10_000.0)
    assert reg["lost_ha"] == 4.0 and reg["gained_ha"] == 0


def test_water_anomalies_are_labelled_as_index_change():
    patches = [(60, 80, 10, 30, 0.2, 0.2),     # NDWI 0.5 -> 0.0 : open water lost
               (82, 98, 40, 56, 0.25, 0.15),   # NDWI 0.5 -> 0.25: index fell, still water
               (10, 30, 60, 80, 0.2, 0.2)]     # outside the polygon: must be ignored
    restore = _fake_pipeline("water", patches)
    try:
        res = ad.run_change_analysis(BBOX, "water", max_cloud=30, threshold=0.2,
                                     min_area_ha=0.5, aoi_geometry=TRIANGLE, **RANGES)
    finally:
        restore()
    assert len(res.anomalies) == 2
    for a in res.anomalies:
        assert a["type"] == "NDWI decrease (index change)"
        assert "Water surface" not in a["type"] + a["reason"]
        assert "not a confirmed change in water extent" in a["reason"]
        assert "estimated open-water extent" in a["reason"]
    lost_block = next(a for a in res.anomalies if a["area_ha"] > 3)
    kept_block = next(a for a in res.anomalies if a["area_ha"] < 3)
    assert abs(lost_block["water_extent"]["lost_ha"] - 4.0) < 1e-6
    assert kept_block["water_extent"]["lost_ha"] == 0
    assert "water mask did not change" in kept_block["reason"]
    # separate, region-independent open-water estimate; outside block excluded by polygon
    assert abs(res.water_summary["lost_ha"] - 4.0) < 1e-6
    # demo water scenario uses the same terminology
    d = ad.run_demo_analysis((30.4, 50.4, 30.45, 50.43), "water", 0.3, 0.05)
    assert d.anomalies and all("NDWI" in a["type"] and "index change" in a["type"]
                               for a in d.anomalies)
    assert d.water_summary is not None


def test_no_old_water_terminology_in_source():
    for path in [*(ROOT / "modules").glob("*.py"), ROOT / "app.py", ROOT / "database.py"]:
        txt = path.read_text()
        assert not re.search(r"Water surface (decrease|increase)", txt), path.name


# --------------------------------------------------------- 4. report wording
def _demo_records():
    db.init_db()
    insp = next(i for i in db.list_inspections() if i["is_demo"])
    return (insp, db.get_anomaly(insp["anomaly_id"]),
            db.get_area(insp["monitoring_area_id"]))


def test_report_wording_does_not_imply_field_confirmation():
    insp, an, area = _demo_records()
    text = rg.deterministic_report(insp, an, area, None)
    ctx = rg.build_context(insp, an, area, None, None)
    assert "SATELLITE-DERIVED INDEX MEASUREMENTS" in text
    assert config.INDEX_DISCLAIMER in text and config.INDEX_DISCLAIMER in ctx
    assert "remotely sensed indicators" in text and "not field-confirmed" in text
    for blob in (text, ctx, gem.REPORT_PROMPT, gem.REPORT_PROMPT + ctx):
        assert "CONFIRMED MEASUREMENT" not in blob.upper()
    for blob in (text, ctx):   # the only "confirmed" allowed is the negated disclaimer
        assert "confirmed" not in blob.replace("not field-confirmed", "").lower()
    p = gem.REPORT_PROMPT
    assert "SATELLITE-DERIVED INDEX MEASUREMENT" in p and "not field-confirmed" in p
    assert "AI-GENERATED HYPOTHESIS" in p and "never describe" in p.lower()
    assert "change in a reflectance index" in p
    # NDWI hypotheses no longer equate the index with confirmed water extent
    assert "not a measure of water quality" in text
    # nothing in the code base still uses the old label
    for path in [*(ROOT / "modules").glob("*.py"), ROOT / "app.py", ROOT / "config.py"]:
        assert "CONFIRMED MEASUREMENT" not in path.read_text().upper(), path.name


def test_pdf_still_builds_with_new_wording():
    from modules import financial_analytics as fin
    insp, an, area = _demo_records()
    text = rg.deterministic_report(insp, an, area, None)
    pdf = rg.build_pdf(insp, an, area, text, None, fin.calculate(2000, 800, 10, 300), "test")
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
