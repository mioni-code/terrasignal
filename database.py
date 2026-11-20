"""SQLite persistence layer. All queries are parameterised."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import config

DEMO_SOURCE = "DEMO — illustrative record, not a satellite observation"

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_areas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    geometry TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT DEFAULT '',
    monitoring_frequency TEXT NOT NULL,
    inspection_budget REAL,
    area_ha REAL DEFAULT 0,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_code TEXT,
    monitoring_area_id INTEGER REFERENCES monitoring_areas(id) ON DELETE SET NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    anomaly_type TEXT NOT NULL,
    anomaly_area REAL,
    ndvi_change REAL,
    ndwi_change REAL,
    priority TEXT NOT NULL,
    priority_score REAL,
    observation_date TEXT,
    baseline_date TEXT,
    cloud_cover REAL,
    reason TEXT,
    source TEXT NOT NULL,
    source_scene_id TEXT,
    polygon TEXT,
    preview_path TEXT,
    status TEXT NOT NULL DEFAULT 'Detected',
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_code TEXT,
    anomaly_id INTEGER REFERENCES anomalies(id) ON DELETE SET NULL,
    monitoring_area_id INTEGER REFERENCES monitoring_areas(id) ON DELETE SET NULL,
    latitude REAL,
    longitude REAL,
    description TEXT,
    priority TEXT DEFAULT 'Attention',
    inspection_status TEXT NOT NULL DEFAULT 'Pending',
    uploaded_image_path TEXT,
    image_is_demo INTEGER DEFAULT 0,
    ai_summary TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS financial_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitoring_area_id INTEGER REFERENCES monitoring_areas(id) ON DELETE SET NULL,
    conventional_cost REAL,
    targeted_cost REAL,
    annual_inspections INTEGER,
    monthly_subscription REAL,
    calculated_savings REAL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitoring_area_id INTEGER REFERENCES monitoring_areas(id) ON DELETE SET NULL,
    analysis_type TEXT,
    observation_date TEXT,
    baseline_period TEXT,
    recent_period TEXT,
    scenes_used INTEGER,
    anomalies_found INTEGER,
    source TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

_INSPECTION_FIELDS = {"inspection_status", "uploaded_image_path", "image_is_demo",
                      "ai_summary", "description", "priority"}
_AREA_FIELDS = {"name", "category", "description", "monitoring_frequency",
                "inspection_budget"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with get_conn() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def init_db(seed: bool = True) -> None:
    with get_conn() as c:
        c.executescript(SCHEMA)
    if seed:
        seed_demo_data()


# ---------------------------------------------------------------- validation
def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_area(name: str, category: str, frequency: str,
                  budget: float | None, geometry: dict | None) -> None:
    _check(bool(name and name.strip()), "Please enter a monitoring area name.")
    _check(len(name) <= 120, "Name is too long (max 120 characters).")
    _check(category in config.CATEGORIES, "Unknown monitoring category.")
    _check(frequency in config.FREQUENCIES, "Unknown monitoring frequency.")
    _check(budget is None or budget >= 0, "Budget cannot be negative.")
    if geometry is not None:
        _check(geometry.get("type") == "Polygon" and geometry.get("coordinates"),
               "Please draw a polygon or rectangle on the map.")


# -------------------------------------------------------------------- areas
def add_area(name: str, geometry: dict, category: str, description: str,
             frequency: str, budget: float | None, area_ha: float,
             is_demo: bool = False) -> int:
    validate_area(name, category, frequency, budget, geometry)
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO monitoring_areas (name, geometry, category, description,"
            " monitoring_frequency, inspection_budget, area_ha, is_demo, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (name.strip(), json.dumps(geometry), category, description.strip(),
             frequency, budget, float(area_ha), int(is_demo), _now()))
        return cur.lastrowid


def update_area(area_id: int, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in _AREA_FIELDS}
    _check(bool(fields), "Nothing to update.")
    current = get_area(area_id)
    _check(current is not None, "Monitoring area not found.")
    merged = {**current, **fields}
    validate_area(merged["name"], merged["category"], merged["monitoring_frequency"],
                  merged["inspection_budget"], None)
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as c:
        c.execute(f"UPDATE monitoring_areas SET {sets} WHERE id = ?",
                  (*fields.values(), area_id))


def delete_area(area_id: int) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM anomalies WHERE monitoring_area_id = ?", (area_id,))
        c.execute("DELETE FROM monitoring_areas WHERE id = ?", (area_id,))


def get_area(area_id: int) -> dict | None:
    rows = query("SELECT * FROM monitoring_areas WHERE id = ?", (area_id,))
    return rows[0] if rows else None


def list_areas() -> list[dict]:
    return query("SELECT * FROM monitoring_areas ORDER BY id")


# ---------------------------------------------------------------- anomalies
def add_anomaly(a: dict) -> int:
    _check(-90 <= a["latitude"] <= 90 and -180 <= a["longitude"] <= 180,
           "Invalid coordinates.")
    _check(a.get("priority") in config.PRIORITIES, "Invalid priority.")
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO anomalies (monitoring_area_id, latitude, longitude,"
            " anomaly_type, anomaly_area, ndvi_change, ndwi_change, priority,"
            " priority_score, observation_date, baseline_date, cloud_cover, reason,"
            " source, source_scene_id, polygon, preview_path, status, is_demo,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (a.get("monitoring_area_id"), a["latitude"], a["longitude"],
             a["anomaly_type"], a.get("anomaly_area"), a.get("ndvi_change"),
             a.get("ndwi_change"), a["priority"], a.get("priority_score"),
             a.get("observation_date"), a.get("baseline_date"), a.get("cloud_cover"),
             a.get("reason"), a["source"], a.get("source_scene_id"),
             json.dumps(a["polygon"]) if a.get("polygon") else None,
             a.get("preview_path"), a.get("status", "Detected"),
             int(a.get("is_demo", False)), _now()))
        aid = cur.lastrowid
        c.execute("UPDATE anomalies SET anomaly_code = ? WHERE id = ?",
                  (f"AN-{aid:05d}", aid))
        return aid


def get_anomaly(anomaly_id: int) -> dict | None:
    rows = query("SELECT * FROM anomalies WHERE id = ?", (anomaly_id,))
    return rows[0] if rows else None


def list_anomalies() -> list[dict]:
    return query("SELECT * FROM anomalies ORDER BY id DESC")


# -------------------------------------------------------------- inspections
def add_inspection(anomaly_id: int | None = None, area_id: int | None = None,
                   latitude: float | None = None, longitude: float | None = None,
                   description: str = "", priority: str = "Attention",
                   is_demo: bool = False) -> int:
    """Create an inspection request (idempotent per anomaly)."""
    if anomaly_id is not None:
        existing = query("SELECT id FROM inspections WHERE anomaly_id = ?",
                         (anomaly_id,))
        if existing:
            return existing[0]["id"]
        an = get_anomaly(anomaly_id)
        _check(an is not None, "Anomaly not found.")
        area_id = area_id or an["monitoring_area_id"]
        latitude, longitude = an["latitude"], an["longitude"]
        description = description or an["reason"] or an["anomaly_type"]
        priority = an["priority"]
        is_demo = bool(an["is_demo"])
    _check(latitude is not None and longitude is not None
           and -90 <= latitude <= 90 and -180 <= longitude <= 180,
           "Valid coordinates are required.")
    _check(priority in config.PRIORITIES, "Invalid priority.")
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO inspections (anomaly_id, monitoring_area_id, latitude,"
            " longitude, description, priority, inspection_status, is_demo,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (anomaly_id, area_id, latitude, longitude, description.strip(), priority,
             "Pending", int(is_demo), _now()))
        iid = cur.lastrowid
        c.execute("UPDATE inspections SET inspection_code = ? WHERE id = ?",
                  (f"INS-{iid:05d}", iid))
        if anomaly_id is not None:
            c.execute("UPDATE anomalies SET status = 'Inspection requested'"
                      " WHERE id = ?", (anomaly_id,))
        return iid


def update_inspection(inspection_id: int, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in _INSPECTION_FIELDS}
    _check(bool(fields), "Nothing to update.")
    if "inspection_status" in fields:
        _check(fields["inspection_status"] in config.INSPECTION_STATUSES,
               "Invalid inspection status.")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as c:
        c.execute(f"UPDATE inspections SET {sets} WHERE id = ?",
                  (*fields.values(), inspection_id))


def get_inspection(inspection_id: int) -> dict | None:
    rows = query("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
    return rows[0] if rows else None


def list_inspections() -> list[dict]:
    return query(
        "SELECT i.*, a.anomaly_code, a.anomaly_type, a.anomaly_area, a.source AS"
        " anomaly_source, m.name AS area_name FROM inspections i"
        " LEFT JOIN anomalies a ON a.id = i.anomaly_id"
        " LEFT JOIN monitoring_areas m ON m.id = i.monitoring_area_id"
        " ORDER BY i.id DESC")


# ----------------------------------------------- observations & finance
def add_observation(area_id: int | None, analysis_type: str, observation_date: str,
                    baseline_period: str, recent_period: str, scenes_used: int,
                    anomalies_found: int, source: str, is_demo: bool) -> int:
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO observations (monitoring_area_id, analysis_type,"
            " observation_date, baseline_period, recent_period, scenes_used,"
            " anomalies_found, source, is_demo, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (area_id, analysis_type, observation_date, baseline_period,
             recent_period, scenes_used, anomalies_found, source, int(is_demo),
             _now()))
        return cur.lastrowid


def list_observations(area_id: int | None = None) -> list[dict]:
    if area_id is None:
        return query("SELECT o.*, m.name AS area_name FROM observations o LEFT JOIN"
                     " monitoring_areas m ON m.id = o.monitoring_area_id"
                     " ORDER BY o.id DESC")
    return query("SELECT * FROM observations WHERE monitoring_area_id = ?"
                 " ORDER BY id DESC", (area_id,))


def add_financial_scenario(area_id: int | None, conventional: float, targeted: float,
                           annual_inspections: int, monthly_subscription: float,
                           savings: float, is_demo: bool = False) -> int:
    _check(min(conventional, targeted, monthly_subscription) >= 0
           and annual_inspections >= 0, "Financial inputs cannot be negative.")
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO financial_scenarios (monitoring_area_id, conventional_cost,"
            " targeted_cost, annual_inspections, monthly_subscription,"
            " calculated_savings, is_demo, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (area_id, conventional, targeted, annual_inspections,
             monthly_subscription, savings, int(is_demo), _now()))
        return cur.lastrowid


def list_scenarios() -> list[dict]:
    return query("SELECT * FROM financial_scenarios ORDER BY id DESC")


# ---------------------------------------------------------------------- KPIs
def get_kpis() -> dict:
    one = lambda sql: query(sql)[0]["v"] or 0  # noqa: E731
    scen = query("SELECT calculated_savings AS s, is_demo FROM financial_scenarios")
    user_scen = [r["s"] for r in scen if not r["is_demo"]]
    return {
        "total_area_ha": one("SELECT SUM(area_ha) AS v FROM monitoring_areas"),
        "zones": one("SELECT COUNT(*) AS v FROM monitoring_areas"),
        "anomalies": one("SELECT COUNT(*) AS v FROM anomalies"),
        "anomalies_real": one("SELECT COUNT(*) AS v FROM anomalies WHERE is_demo = 0"),
        "pending": one("SELECT COUNT(*) AS v FROM inspections WHERE inspection_status"
                       " IN ('Pending','Ready for review')"),
        "completed": one("SELECT COUNT(*) AS v FROM inspections WHERE"
                         " inspection_status IN ('Analysed','Report generated')"),
        "savings": sum(user_scen) if user_scen else None,
        "savings_demo": sum(r["s"] for r in scen if r["is_demo"]) if scen else None,
    }


# ---------------------------------------------------------------------- seed
def seed_demo_data() -> None:
    """Insert clearly labelled example records once (empty DB only)."""
    if query("SELECT COUNT(*) AS v FROM monitoring_areas")[0]["v"] > 0:
        return

    def rect(lon, lat, dx, dy):
        return {"type": "Polygon", "coordinates": [[
            [lon, lat], [lon + dx, lat], [lon + dx, lat + dy],
            [lon, lat + dy], [lon, lat]]]}

    areas = [
        ("[DEMO] Reservoir shoreline", rect(31.40, 49.72, 0.03, 0.02),
         "Water bodies", "Example record for demonstration only.", "Monthly", 2500),
        ("[DEMO] Forest block", rect(30.20, 50.50, 0.03, 0.02),
         "Forests", "Example record for demonstration only.", "Bi-weekly", None),
        ("[DEMO] Agricultural field", rect(30.90, 50.00, 0.025, 0.015),
         "Agricultural land", "Example record for demonstration only.", "Monthly", 1200),
    ]
    ids = []
    for n, g, cat, desc, freq, bud in areas:
        ha = _approx_ha(g)
        ids.append(add_area(n, g, cat, desc, freq, bud, ha, is_demo=True))

    for aid_, atype_ in zip(ids, ["water", "vegetation", "vegetation"]):
        add_observation(aid_, atype_, "", "DEMO", "DEMO", 0, 1, DEMO_SOURCE, True)
    demo_anoms = [
        (ids[1], 50.508, 30.215, "Vegetation decrease", 2.4, -0.38, None, "High", 0.74),
        (ids[2], 50.006, 30.912, "Vegetation decrease", 0.9, -0.27, None, "Attention", 0.42),
        (ids[0], 49.728, 31.415, "NDWI decrease (index change)", 1.6, None, -0.41, "Attention", 0.51),
    ]
    for aid, lat, lon, typ, ha, dn, dw, pr, sc in demo_anoms:
        an = add_anomaly({
            "monitoring_area_id": aid, "latitude": lat, "longitude": lon,
            "anomaly_type": typ, "anomaly_area": ha, "ndvi_change": dn,
            "ndwi_change": dw, "priority": pr, "priority_score": sc,
            "observation_date": None, "source": DEMO_SOURCE,
            "reason": "[DEMO] Illustrative example record.", "is_demo": True})
    first = add_inspection(anomaly_id=an, is_demo=True)
    update_inspection(first, inspection_status="Pending")
    add_financial_scenario(None, 2000, 800, 10, 300, 1200 * 10 - 300 * 12, is_demo=True)


def _approx_ha(geom: dict) -> float:
    """Planar-approximation area (hectares) of a lon/lat polygon."""
    import math
    ring = geom["coordinates"][0]
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx = 111_320 * math.cos(math.radians(lat0))
    ky = 110_540
    pts = [(p[0] * kx, p[1] * ky) for p in ring]
    s = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return abs(s) / 2 / 10_000
