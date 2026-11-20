"""Core tests (no network, no Streamlit). Run: python tests/test_core.py  (or pytest)."""
from __future__ import annotations

import io
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "test.db"
import database as db  # noqa: E402
from modules import (anomaly_detection as ad, drone_simulation as drone,  # noqa: E402
                     financial_analytics as fin, report_generator as rg,
                     satellite_data as sat, vegetation_analysis as veg,
                     water_analysis as wat)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def test_finance():
    r = fin.calculate(2000, 800, 10, 300)
    assert r.savings_per_inspection == 1200
    assert r.annual_gross_savings == 12000
    assert r.annual_platform_cost == 3600
    assert r.annual_net_benefit == 8400
    assert abs(r.roi_percent - 233.3333333) < 1e-4
    assert r.break_even_inspections == 3
    # break-even rounds up
    assert fin.calculate(1000, 0, 5, 250).break_even_inspections == 3  # 3000/1000
    assert fin.calculate(1000, 400, 5, 250).break_even_inspections == 5  # 3000/600 = 5
    # zero platform cost -> ROI undefined, break-even 0
    z = fin.calculate(1000, 400, 5, 0)
    assert z.roi_percent is None and z.break_even_inspections == 0
    # no savings -> never breaks even, negative net
    neg = fin.calculate(500, 800, 10, 100)
    assert neg.savings_per_inspection == -300 and neg.break_even_inspections is None
    assert neg.annual_net_benefit == -3000 - 1200 and neg.roi_percent < 0
    # zero inspections
    assert fin.calculate(1000, 400, 0, 100).annual_net_benefit == -1200
    raises(ValueError, fin.calculate, -1, 0, 1, 1)
    raises(ValueError, fin.calculate, 1, 0, -1, 1)


def test_indices():
    red = np.array([[0.1, 0.2]], dtype="float32")
    nir = np.array([[0.5, 0.2]], dtype="float32")
    ndvi = veg.compute_ndvi(red, nir)
    assert abs(ndvi[0, 0] - (0.4 / 0.6)) < 1e-6 and ndvi[0, 1] == 0
    assert np.isnan(veg.compute_ndvi(np.zeros((1, 1)), np.zeros((1, 1)))[0, 0])
    g = np.array([[0.3]], dtype="float32")
    n = np.array([[0.1]], dtype="float32")
    assert abs(wat.compute_ndwi(g, n)[0, 0] - 0.5) < 1e-6
    scl = np.array([[4, 5, 6, 8, 9, 3, 0, 10]])
    assert veg.scl_valid_mask(scl).tolist() == [[True, True, True, False, False, False, False, False]]
    masked = veg.masked_index(np.ones((1, 8), dtype="float32"), scl)
    assert np.isfinite(masked).sum() == 3
    comp = veg.composite([np.array([[1.0, np.nan]]), np.array([[3.0, np.nan]]), np.array([[5.0, 2.0]])])
    assert comp[0, 0] == 3.0 and comp[0, 1] == 2.0


def test_detection():
    ch = np.zeros((100, 100), dtype="float32")
    ch[10:30, 10:30] = -0.5          # 400 px big region
    ch[60:62, 60:62] = -0.6          # 4 px tiny region (noise)
    ch[70:80, 70:80] = +0.5          # increase: ignored for vegetation
    ch[5, 90] = np.nan
    ident = lambda c, r: (30.0 + c * 1e-4, 50.0 - r * 1e-4)  # noqa: E731
    regs = ad.detect_regions(ch, 0.2, "decrease", 100.0, ident, min_area_ha=0.5)
    assert len(regs) == 1, regs                      # tiny region filtered
    r = regs[0]
    assert abs(r["area_ha"] - 4.0) < 1e-6            # 400 px * 100 m2 = 4 ha
    assert abs(r["mean_change"] + 0.5) < 1e-6
    assert r["priority"] in config.PRIORITIES
    assert regs[0]["polygon"]["coordinates"][0][0] == regs[0]["polygon"]["coordinates"][0][-1]
    inc = ad.detect_regions(ch, 0.2, "increase", 100.0, ident, 0.5)
    assert len(inc) == 1 and inc[0]["mean_change"] > 0
    # threshold respected
    assert ad.detect_regions(ch, 0.6, "decrease", 100.0, ident, 0.01) != []  # 4px region at -0.6
    assert ad.detect_regions(np.zeros((10, 10)), 0.2, "decrease", 100.0, ident) == []
    raises(ValueError, ad.detect_regions, ch, 0.2, "sideways", 100.0, ident)
    # priority tiers
    assert ad.priority_level(0.8) == "High" and ad.priority_level(0.4) == "Attention"
    assert ad.priority_level(0.1) == "Normal"


def test_validation():
    raises(sat.SatelliteError, sat.validate_request, None, [])
    from datetime import date, timedelta
    ok_bbox = (30.0, 50.0, 30.05, 50.03)
    raises(sat.SatelliteError, sat.validate_request, ok_bbox,
           [(date(2025, 5, 2), date(2025, 5, 1))])
    raises(sat.SatelliteError, sat.validate_request, ok_bbox,
           [(date.today(), date.today() + timedelta(days=5))])
    raises(sat.SatelliteError, sat.validate_request, (30.0, 50.0, 32.0, 52.0),
           [(date(2025, 1, 1), date(2025, 2, 1))])  # too large
    sat.validate_request(ok_bbox, [(date(2025, 1, 1), date(2025, 2, 1))])


def test_database_and_labels():
    db.init_db()
    areas = db.list_areas()
    assert areas and all(a["is_demo"] for a in areas)
    assert all(a["name"].startswith("[DEMO]") for a in areas)
    assert all(a["is_demo"] and "DEMO" in a["source"] for a in db.list_anomalies())
    k = db.get_kpis()
    assert k["anomalies_real"] == 0 and k["savings"] is None   # demo never counted as real

    geom = {"type": "Polygon", "coordinates": [[[30, 50], [30.01, 50], [30.01, 50.01], [30, 50.01], [30, 50]]]}
    raises(ValueError, db.add_area, "", geom, "Forests", "", "Monthly", None, 1)
    raises(ValueError, db.add_area, "x", geom, "Nope", "", "Monthly", None, 1)
    raises(ValueError, db.add_area, "x", geom, "Forests", "", "Monthly", -5, 1)
    aid = db.add_area("My forest'; DROP TABLE monitoring_areas;--", geom, "Forests", "d", "Monthly", 100, 80)
    assert db.get_area(aid)["name"].startswith("My forest")   # injection is inert
    db.update_area(aid, name="Renamed", category="Riverbanks")
    assert db.get_area(aid)["name"] == "Renamed"
    raises(ValueError, db.update_area, aid, category="bad")

    an = db.add_anomaly({"monitoring_area_id": aid, "latitude": 50.0, "longitude": 30.0,
                         "anomaly_type": "Vegetation decrease", "anomaly_area": 2.0,
                         "ndvi_change": -0.3, "priority": "High", "priority_score": 0.7,
                         "observation_date": "2026-10-01", "source": "Sentinel-2 L2A", "is_demo": False})
    assert db.get_anomaly(an)["anomaly_code"] == f"AN-{an:05d}"
    iid = db.add_inspection(anomaly_id=an)
    assert db.add_inspection(anomaly_id=an) == iid            # idempotent
    assert db.get_anomaly(an)["status"] == "Inspection requested"
    raises(ValueError, db.update_inspection, iid, inspection_status="Flying")
    for s in config.INSPECTION_STATUSES:
        db.update_inspection(iid, inspection_status=s)
    biz = db.add_inspection(area_id=aid, latitude=50.1, longitude=30.1, description="check outfall")
    assert db.get_inspection(biz)["anomaly_id"] is None
    raises(ValueError, db.add_inspection, None, aid, 123.0, 30.0, "bad lat")
    assert db.get_kpis()["anomalies_real"] == 1

    # persistence: new connection still sees the data
    assert db.get_area(aid) is not None and db.get_inspection(iid) is not None
    db.delete_area(aid)
    assert db.get_area(aid) is None and db.get_anomaly(an) is None
    assert db.get_inspection(iid)["anomaly_id"] is None       # inspection kept, link cleared


def test_demo_analysis_labelled():
    res = ad.run_demo_analysis((30.4, 50.4, 30.45, 50.43), "vegetation", 0.2, 0.05)
    assert res.mode == "DEMO" and "DEMO" in res.source and res.anomalies
    assert all("type" in a and "reason" in a for a in res.anomalies)
    w = ad.run_demo_analysis((30.4, 50.4, 30.45, 50.43), "water", 0.3, 0.05)
    assert w.index_name == "NDWI"


def test_upload_validation_and_pdf():
    raises(ValueError, drone.validate_and_save_upload, "x.exe", b"MZ")
    raises(ValueError, drone.validate_and_save_upload, "x.png", b"not an image")
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 30)).save(b, "PNG")
    p = drone.validate_and_save_upload("ok.png", b.getvalue())
    assert p.exists()

    insp_id = next(i["id"] for i in db.list_inspections() if i["is_demo"])
    drone.attach_image(insp_id, drone.ensure_demo_image(), True)
    insp = db.get_inspection(insp_id)
    assert insp["inspection_status"] == "Ready for review"
    an = db.get_anomaly(insp["anomaly_id"])
    area = db.get_area(insp["monitoring_area_id"])
    text = rg.deterministic_report(insp, an, area, None)
    assert "DEMO record" in text and "SATELLITE-DERIVED INDEX MEASUREMENTS" in text
    pdf = rg.build_pdf(insp, an, area, text, None, fin.calculate(2000, 800, 10, 300), "deterministic")
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
    assert "SIMULATED" in drone.SIM_LABEL


def test_gemini_absent_is_graceful():
    import os
    from modules import gemini_service as gem
    os.environ.pop("GEMINI_API_KEY", None)
    ok, _ = gem.is_available()
    assert gem.analyze_image(b"x")[0] is None and gem.generate_report("x")[0] is None
    assert ok is False or gem.api_key()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
