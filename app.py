"""TerraSignal — Streamlit entry point.  Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import config
import database as db
from modules import (anomaly_detection as ad, drone_simulation as drone,
                     financial_analytics as fin, gemini_service as gem,
                     map_interface as mi, monitoring_areas as ma,
                     report_generator as rg, satellite_data as sat, ui)

st.set_page_config(page_title="TerraSignal", page_icon="🌿", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
db.init_db()

PAGES = ["Overview", "Satellite Scan", "Monitoring Areas", "Drone Verification",
         "Reports & Finance"]


# ------------------------------------------------------------ small helpers
def show_map(m, key: str, height: int = 460, returned=None):
    """st_folium wrapper tolerant to API changes between versions."""
    try:
        from streamlit_folium import st_folium
    except Exception:
        st.error("streamlit-folium is not installed. Run: pip install -r requirements.txt")
        return {}
    kw = dict(key=key, height=height, returned_objects=returned if returned is not None else [])
    try:
        return st_folium(m, use_container_width=True, **kw)
    except TypeError:
        return st_folium(m, width=1100, **kw)


def table(df: pd.DataFrame):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True)


def chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=600, show_spinner=False)
def _sat_status():
    return sat.check_connection()


def goto(page: str, **state):
    st.session_state["nav"] = page
    st.session_state.update(state)


def fmt_range(r: tuple[date, date]) -> str:
    return f"{r[0]} → {r[1]}"


# ------------------------------------------------------------------ sidebar
def sidebar() -> str:
    st.sidebar.markdown("<div class='ts-brand'>🌿 TerraSignal</div>"
                        f"<div class='ts-tag'>{config.TAGLINE}</div>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    sat_ok, sat_msg = _sat_status()
    gem_ok, gem_msg = gem.is_available()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='ts-status'>" + ui.status_dot(sat_ok, "Satellite data", sat_msg)
        + ui.status_dot(gem_ok, "Gemini AI", gem_msg + ("" if gem_ok else " (optional)"))
        + "</div>", unsafe_allow_html=True)
    st.sidebar.caption("Prototype for demonstration. Not an official environmental "
                       "monitoring system.")
    return page


# ----------------------------------------------------------------- overview
def page_overview():
    st.title("Overview")
    k = db.get_kpis()
    cols = st.columns(6)
    items = [
        ("Monitored area", f"{k['total_area_ha']:,.0f} ha", "incl. DEMO example areas"),
        ("Monitoring zones", str(k["zones"]), ""),
        ("Detected anomalies", str(k["anomalies"]),
         f"{k['anomalies_real']} from real satellite data"),
        ("Pending inspections", str(k["pending"]), ""),
        ("Completed inspections", str(k["completed"]), ""),
        ("Est. inspection savings",
         "—" if k["savings"] is None else f"{k['savings']:,.0f}",
         "your saved scenarios" if k["savings"] is not None
         else "save a scenario in Reports & Finance"),
    ]
    for c, (a, b, s) in zip(cols, items):
        c.markdown(ui.kpi(a, b, s), unsafe_allow_html=True)
    st.caption("Records labelled DEMO are seeded examples, not satellite observations. "
               "Financial figures are illustrative.")

    areas, anoms = db.list_areas(), db.list_anomalies()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Monitoring areas and detected anomalies")
        show_map(mi.overview_map(areas, anoms), "ov_map", 430)
    with right:
        st.subheader("Anomaly priority distribution")
        if anoms:
            import plotly.express as px
            df = pd.DataFrame(anoms)
            df["Priority"] = df["priority"].map(config.PRIORITY_LABELS)
            cnt = df.groupby("Priority").size().reset_index(name="Count")
            fig = px.bar(cnt, x="Priority", y="Count", color="Priority",
                         color_discrete_map={config.PRIORITY_LABELS[p]: config.COLORS[p]
                                             for p in config.PRIORITIES})
            fig.update_layout(showlegend=False, height=330, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            chart(fig)
        else:
            st.info("No anomalies yet.")

    a, b = st.columns(2)
    with a:
        st.subheader("Latest alerts")
        if anoms:
            for an in anoms[:5]:
                st.markdown(
                    f"<div class='ts-card'>{ui.badge(an['priority'], bool(an['is_demo']))} "
                    f"<b>{an['anomaly_code']}</b> — {an['anomaly_type']}, "
                    f"{an['anomaly_area'] or 0:.1f} ha<br><span class='ts-kpi-sub'>"
                    f"{an['latitude']:.4f}, {an['longitude']:.4f} · "
                    f"{an['observation_date'] or 'no observation date'}</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("No alerts yet. Run a satellite scan.")
    with b:
        st.subheader("Recent satellite observations")
        obs = db.list_observations()
        if obs:
            df = pd.DataFrame(obs)[["created_at", "area_name", "analysis_type",
                                    "anomalies_found", "is_demo"]]
            df["is_demo"] = df["is_demo"].map({1: "DEMO", 0: "Real"})
            df.columns = ["Run at", "Area", "Type", "Anomalies", "Data"]
            table(df.head(8))
        else:
            st.info("No observations yet.")
        st.subheader("Monitoring activity")
        if obs:
            import plotly.express as px
            df = pd.DataFrame(obs)
            df["Month"] = df["created_at"].str[:7]
            cnt = df.groupby("Month").size().reset_index(name="Analyses run")
            fig = px.bar(cnt, x="Month", y="Analyses run",
                         color_discrete_sequence=[config.COLORS["primary"]])
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            chart(fig)


# ----------------------------------------------------------- satellite scan
def _preview_for(res, region) -> str | None:
    """Save a small preview crop of the recent image around a region."""
    from PIL import Image, ImageDraw
    import numpy as np
    r0, r1, c0, c1 = region["bbox_px"]
    pad = 15
    if res.cir_recent is not None:
        img = res.cir_recent
    else:
        rgba = mi.index_rgba(res.recent_index, res.index_name)
        img = rgba[..., :3]
    H, W = img.shape[:2]
    crop = img[max(0, r0 - pad):min(H, r1 + pad), max(0, c0 - pad):min(W, c1 + pad)]
    pil = Image.fromarray(np.ascontiguousarray(crop)).resize(
        (max(200, crop.shape[1] * 3), max(200, crop.shape[0] * 3)), Image.NEAREST)
    if res.mode == "DEMO":
        ImageDraw.Draw(pil).text((6, 6), "DEMO - SYNTHETIC", fill=(255, 255, 255))
    name = hashlib.md5(f"{res.source}{r0}{c0}{r1}{c1}{id(res)}".encode()).hexdigest()[:12]
    path = config.PREVIEW_DIR / f"{name}.png"
    pil.save(path)
    return str(path)


def _persist(res, area_id: int | None) -> list[int]:
    demo = res.mode == "DEMO"
    rec_date = max((s["date"] for s in res.recent_scenes), default=None)
    base_date = max((s["date"] for s in res.baseline_scenes), default=None)
    clouds = [s["cloud"] for s in res.recent_scenes]
    cloud = sum(clouds) / len(clouds) if clouds else None
    ids = []
    for r in res.anomalies:
        ids.append(db.add_anomaly({
            "monitoring_area_id": area_id, "latitude": r["latitude"],
            "longitude": r["longitude"], "anomaly_type": r["type"],
            "anomaly_area": r["area_ha"],
            "ndvi_change": r["mean_change"] if res.index_name == "NDVI" else None,
            "ndwi_change": r["mean_change"] if res.index_name == "NDWI" else None,
            "priority": r["priority"], "priority_score": r["priority_score"],
            "observation_date": str(rec_date) if rec_date else None,
            "baseline_date": str(base_date) if base_date else None,
            "cloud_cover": cloud, "reason": r["reason"],
            "source": res.source,
            "source_scene_id": ", ".join(s["id"] for s in res.recent_scenes) or None,
            "polygon": r["polygon"], "preview_path": _preview_for(res, r),
            "is_demo": demo}))
    db.add_observation(
        area_id, res.analysis_type, str(rec_date) if rec_date else "",
        fmt_range(res.params["baseline_range"]) if "baseline_range" in res.params else "DEMO",
        fmt_range(res.params["recent_range"]) if "recent_range" in res.params else "DEMO",
        len(res.recent_scenes) + len(res.baseline_scenes), len(res.anomalies),
        res.source, demo)
    return ids


def _date_range(label: str, default: tuple[date, date], key: str):
    v = st.date_input(label, value=default, key=key, max_value=date.today())
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (v[0], v[1])
    return None


def page_scan():
    st.title("Satellite Scan")
    st.caption("Macro view: compare Sentinel-2 imagery from two periods to find areas "
               "that changed and deserve a closer look.")
    areas = db.list_areas()
    pref = st.session_state.get("scan_area_pref")
    mode = st.radio("Area of Interest", ["Draw on map", "Use a saved monitoring area"],
                    horizontal=True, key="scan_src",
                    index=1 if pref else 0)
    geom, area_id = None, None
    if mode == "Draw on map":
        st.caption("Use the rectangle or polygon tool (left of the map). Keep it below "
                   f"{config.MAX_AOI_KM2:.0f} km².")
        out = show_map(mi.draw_map(zoom=6), "scan_draw", 420,
                       ["all_drawings", "last_active_drawing"])
        geom = mi.extract_geometry(out) or st.session_state.get("scan_geom")
        if geom:
            st.session_state["scan_geom"] = geom
    else:
        if not areas:
            st.info("No saved monitoring areas yet. Add one in 'Monitoring Areas'.")
        else:
            labels = {a["id"]: a["name"] for a in areas}
            ids = list(labels)
            idx = ids.index(pref) if pref in ids else 0
            area_id = st.selectbox("Monitoring area", ids, index=idx,
                                   format_func=labels.get)
            geom = ma.area_geometry(next(a for a in areas if a["id"] == area_id))
            show_map(mi.draw_map(center=mi.geometry_center(geom), zoom=13,
                                 highlight_geom=geom), "scan_area_map", 320)
    bbox = mi.geometry_bbox(geom) if geom else None
    if bbox:
        st.caption(f"AOI bounding box: W {bbox[0]:.4f}, S {bbox[1]:.4f}, E {bbox[2]:.4f}, "
                   f"N {bbox[3]:.4f} · approx. {sat.bbox_area_km2(bbox):.1f} km²")

    c1, c2 = st.columns(2)
    with c1:
        atype = st.radio("Analysis type", ["Vegetation monitoring", "Water surface monitoring"],
                         key="scan_type")
        kind = "vegetation" if atype.startswith("Veg") else "water"
        today = date.today()
        rec = _date_range("Recent observation period", (today - timedelta(days=60), today), "rec")
        base = _date_range("Baseline period (default: same season, one year earlier)",
                           (today - timedelta(days=425), today - timedelta(days=365)), "base")
    with c2:
        cloud = st.slider("Maximum cloud cover (%)", 0, 100, 30)
        default_thr = config.DEFAULT_NDVI_THRESHOLD if kind == "vegetation" else config.DEFAULT_NDWI_THRESHOLD
        thr = st.slider(f"Anomaly threshold (|{'NDVI' if kind == 'vegetation' else 'NDWI'} change|)",
                        0.05, 0.80, default_thr, 0.01, key=f"thr_{kind}")
        min_ha = st.slider("Minimum region size (ha)", 0.1, 5.0, config.DEFAULT_MIN_REGION_HA, 0.1)
        if kind == "water":
            st.caption(wat_note())

    run = st.button("Run Satellite Analysis", type="primary")
    if run:
        st.session_state.pop("scan_error", None)
        if rec is None or base is None:
            st.session_state["scan_error"] = "Please select a complete start and end date for both periods."
        else:
            with st.spinner("Searching Sentinel-2 and reading the AOI window…"):
                try:
                    res = ad.run_change_analysis(bbox, kind, base, rec, cloud, thr, min_ha,
                                                 aoi_geometry=geom)
                    st.session_state["scan_result"] = res
                    st.session_state["scan_ids"] = _persist(res, area_id)
                    st.session_state["scan_area_id"] = area_id
                except sat.SatelliteError as exc:
                    st.session_state["scan_error"] = str(exc)
                except Exception as exc:  # never crash the app
                    st.session_state["scan_error"] = (
                        f"Unexpected problem during analysis ({type(exc).__name__}). "
                        "Try a smaller area or different dates.")

    if st.session_state.get("scan_error"):
        st.error(st.session_state["scan_error"])
        st.markdown(ui.banner("demo", "Real satellite analysis did not complete. You can "
                                      "explore the workflow with DEMO MODE (synthetic data, "
                                      "clearly labelled)."), unsafe_allow_html=True)

    with st.expander("Demo mode (synthetic data — not satellite observations)"):
        st.write("Generates artificial arrays so the workflow can be demonstrated offline. "
                 "All outputs are labelled DEMO and are never mixed with real observations.")
        if st.button("Load DEMO scenario"):
            demo_bbox = bbox or (30.40, 50.40, 30.45, 50.43)
            res = ad.run_demo_analysis(demo_bbox, kind, thr, min_ha,
                                       aoi_geometry=geom if bbox else None)
            st.session_state["scan_result"] = res
            st.session_state["scan_ids"] = _persist(res, None)
            st.session_state["scan_area_id"] = None
            st.session_state.pop("scan_error", None)

    res = st.session_state.get("scan_result")
    if res is not None:
        render_result(res)


def wat_note() -> str:
    from modules.water_analysis import NDWI_NOTE
    return NDWI_NOTE


def render_result(res):
    st.markdown("---")
    st.header("Results")
    if res.mode == "DEMO":
        st.markdown(ui.banner("demo", "DEMO MODE — SYNTHETIC DATA. These are not satellite "
                                      "observations."), unsafe_allow_html=True)
    else:
        st.markdown(ui.banner("info", f"Real data: {res.source}. Tile {res.params.get('tile')}."),
                    unsafe_allow_html=True)
    for n in res.notes:
        if res.mode != "DEMO":
            st.warning(n)

    if res.mode == "REAL":
        rows = ([{"Period": "Baseline", "Acquired": str(s["date"]), "Cloud %": round(s["cloud"], 1),
                  "Scene": s["id"]} for s in res.baseline_scenes]
                + [{"Period": "Recent", "Acquired": str(s["date"]), "Cloud %": round(s["cloud"], 1),
                    "Scene": s["id"]} for s in res.recent_scenes])
        st.subheader("Scenes used (actual acquisition dates)")
        table(pd.DataFrame(rows))

    st.subheader("Baseline, recent and change")
    cols = st.columns(2)
    if res.cir_baseline is not None:
        cols[0].image(res.cir_baseline, caption="Baseline image (colour-infrared: NIR/Red/Green)")
        cols[1].image(res.cir_recent, caption="Recent image (colour-infrared)")
    cols = st.columns(2)
    cols[0].image(mi.index_rgba(res.recent_index, res.index_name),
                  caption=f"Recent {res.index_name}" + (" (DEMO)" if res.mode == "DEMO" else ""))
    cols[1].image(mi.change_rgba(res.change),
                  caption=f"{res.index_name} change (recent − baseline)"
                          + (" (DEMO)" if res.mode == "DEMO" else ""))
    st.markdown(mi.legend_html(res.index_name), unsafe_allow_html=True)
    if res.water_summary:
        w = res.water_summary
        st.info("**Estimated open-water extent (NDWI > 0 mask, remote estimate):** baseline "
                f"{w['baseline_water_ha']:.1f} ha → recent {w['recent_water_ha']:.1f} ha "
                f"(lost {w['lost_ha']:.1f} ha, gained {w['gained_ha']:.1f} ha). This is "
                "reported separately from the NDWI index change shown above. " + wat_note())
    if res.aoi_geometry is not None and res.aoi_mask is not None:
        st.caption(f"Polygon mask applied: {res.aoi_mask.sum():,} of {res.aoi_mask.size:,} "
                   f"pixels ({res.aoi_mask.mean():.0%} of the bounding rectangle) were "
                   "analysed; pixels outside your polygon were excluded.")

    st.subheader("Change detection map and anomaly outlines")
    show_map(mi.result_map(res), "result_map", 520)

    ids = st.session_state.get("scan_ids", [])
    if not res.anomalies:
        st.success("No regions exceeded the threshold and minimum size. Try a lower "
                   "threshold or different dates.")
        return
    st.subheader(f"Inspection points ({len(res.anomalies)})")
    st.caption("Priority is a heuristic preliminary screening indicator (change magnitude "
               "and area), not an AI confidence score.")
    saved = [db.get_anomaly(i) for i in ids]
    df = pd.DataFrame([{
        "ID": a["anomaly_code"], "Lat": round(a["latitude"], 5), "Lon": round(a["longitude"], 5),
        "Type": a["anomaly_type"], "Area (ha)": a["anomaly_area"],
        "Change": a["ndvi_change"] if a["ndvi_change"] is not None else a["ndwi_change"],
        "Priority": config.PRIORITY_LABELS[a["priority"]], "Score": a["priority_score"],
        "Observed": a["observation_date"] or "DEMO"} for a in saved])
    table(df)

    st.caption("Click a marker to select it, or use the selector below.")
    out = show_map(mi.anomaly_map(saved, zoom=13), "pts_map", 360, ["last_object_clicked_tooltip"])
    codes = [a["anomaly_code"] for a in saved]
    clicked = (out or {}).get("last_object_clicked_tooltip")
    if clicked in codes and st.session_state.get("_last_click") != clicked:
        st.session_state["_last_click"] = clicked
        st.session_state["scan_sel"] = clicked
    sel = st.selectbox("Anomaly details", codes, key="scan_sel")
    an = saved[codes.index(sel)]
    show_anomaly_details(an)


def show_anomaly_details(an: dict):
    st.markdown(f"#### {an['anomaly_code']} {ui.badge(an['priority'], bool(an['is_demo']))}",
                unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        if an.get("preview_path"):
            st.image(an["preview_path"], caption="Source satellite image crop"
                     + (" (DEMO)" if an["is_demo"] else ""))
    with c2:
        st.write(f"**Coordinates:** {an['latitude']:.5f}, {an['longitude']:.5f}")
        st.write(f"**Type:** {an['anomaly_type']} · **Area:** {an['anomaly_area'] or 0:.2f} ha")
        st.write(f"**Detection date:** {an['observation_date'] or 'n/a'} "
                 f"(baseline {an['baseline_date'] or 'n/a'})")
        st.write(f"**Priority:** {config.PRIORITY_LABELS[an['priority']]} "
                 f"(heuristic score {an['priority_score']})")
        st.write(f"**Reason:** {an['reason']}")
        st.write(f"**Source:** {an['source']}")
    existing = db.query("SELECT id FROM inspections WHERE anomaly_id = ?", (an["id"],))
    if existing:
        st.success("An inspection request already exists for this anomaly. "
                   "Open 'Drone Verification' to continue.")
    elif st.button("Request Detailed Inspection", type="primary", key=f"req_{an['id']}"):
        try:
            iid = db.add_inspection(anomaly_id=an["id"])
            st.success(f"Inspection request {db.get_inspection(iid)['inspection_code']} created "
                       "and saved. (No physical drone is dispatched.)")
        except ValueError as exc:
            st.error(str(exc))


# ---------------------------------------------------------- monitoring areas
def page_areas():
    st.title("Monitoring Areas")
    st.info(ma.frequency_note())
    areas = db.list_areas()
    show_map(mi.overview_map(areas, []), "areas_map", 360)

    st.subheader("Add Monitoring Area")
    st.caption("Draw a polygon or rectangle, then fill in the details.")
    out = show_map(mi.draw_map(zoom=6), "area_draw", 380, ["all_drawings", "last_active_drawing"])
    geom = mi.extract_geometry(out) or st.session_state.get("area_geom")
    if geom:
        st.session_state["area_geom"] = geom
        st.caption(f"Drawn shape: approx. {ma.polygon_area_ha(geom):,.1f} ha")
    with st.form("add_area", clear_on_submit=False):
        name = st.text_input("Monitoring area name")
        c1, c2, c3 = st.columns(3)
        cat = c1.selectbox("Category", config.CATEGORIES)
        freq = c2.selectbox("Monitoring frequency", config.FREQUENCIES, index=2)
        budget = c3.number_input("Inspection budget (optional)", min_value=0.0, value=0.0, step=100.0)
        desc = st.text_area("Description")
        ok = st.form_submit_button("Save monitoring area", type="primary")
    if ok:
        try:
            ma.create_area(name, st.session_state.get("area_geom"), cat, desc, freq,
                           budget if budget > 0 else None)
            st.session_state.pop("area_geom", None)
            st.success("Monitoring area saved.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Saved areas")
    if not areas:
        st.info("No monitoring areas yet.")
        return
    df = pd.DataFrame(areas)
    df["Name"] = df.apply(lambda r: r["name"] + (" [DEMO]" if r["is_demo"] else ""), axis=1)
    table(df[["id", "Name", "category", "monitoring_frequency", "area_ha", "inspection_budget",
              "created_at"]].rename(columns={"monitoring_frequency": "frequency",
                                             "area_ha": "area (ha)", "inspection_budget": "budget"}))
    labels = {a["id"]: a["name"] for a in areas}
    aid = st.selectbox("Select an area to manage", list(labels), format_func=labels.get)
    a = db.get_area(aid)
    with st.form(f"edit_{aid}"):
        n = st.text_input("Name", a["name"])
        c1, c2, c3 = st.columns(3)
        cat = c1.selectbox("Category", config.CATEGORIES, index=config.CATEGORIES.index(a["category"]))
        fr = c2.selectbox("Frequency", config.FREQUENCIES,
                          index=config.FREQUENCIES.index(a["monitoring_frequency"]))
        bud = c3.number_input("Budget (optional)", min_value=0.0, value=float(a["inspection_budget"] or 0))
        d = st.text_area("Description", a["description"] or "")
        save = st.form_submit_button("Save changes")
    if save:
        try:
            db.update_area(aid, name=n, category=cat, monitoring_frequency=fr,
                           inspection_budget=bud if bud > 0 else None, description=d)
            st.success("Updated.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Run satellite analysis", on_click=goto, args=("Satellite Scan",),
              kwargs={"scan_area_pref": aid, "scan_src": "Use a saved monitoring area"},
              type="primary")
    if b2.button("Delete area"):
        st.session_state["confirm_del"] = aid
    if st.session_state.get("confirm_del") == aid:
        st.warning("Delete this area and its detected anomalies? This cannot be undone.")
        if st.button("Yes, delete permanently"):
            db.delete_area(aid)
            st.session_state.pop("confirm_del", None)
            st.rerun()
    st.subheader("Previous observations")
    obs = db.list_observations(aid)
    if obs:
        df = pd.DataFrame(obs)[["created_at", "analysis_type", "observation_date",
                                "baseline_period", "recent_period", "anomalies_found", "is_demo"]]
        df["is_demo"] = df["is_demo"].map({1: "DEMO", 0: "Real"})
        table(df)
    else:
        st.caption("No analyses have been run for this area yet.")


# --------------------------------------------------------- drone verification
def page_drone():
    st.title("Drone Verification")
    st.markdown(ui.banner("sim", drone.SIM_LABEL), unsafe_allow_html=True)
    st.caption("Demonstration of how a satellite alert leads to a detailed local inspection. "
               "No drone is controlled or dispatched.")
    insps = db.list_inspections()
    with st.expander("Add a business-requested inspection point (no satellite anomaly needed)"):
        areas = db.list_areas()
        if not areas:
            st.info("Create a monitoring area first.")
        else:
            labels = {a["id"]: a["name"] for a in areas}
            aid = st.selectbox("Monitoring area", list(labels), format_func=labels.get, key="bi_area")
            clat, clon = ma.area_centroid(next(a for a in areas if a["id"] == aid))
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=float(clat), format="%.6f", key=f"bi_lat_{aid}")
            lon = c2.number_input("Longitude", value=float(clon), format="%.6f", key=f"bi_lon_{aid}")
            d = st.text_input("What should be inspected?", key="bi_desc")
            pr = st.selectbox("Priority", config.PRIORITIES, index=1, format_func=config.PRIORITY_LABELS.get)
            if st.button("Add inspection point"):
                try:
                    db.add_inspection(area_id=aid, latitude=lat, longitude=lon,
                                      description=d or "Business-requested inspection", priority=pr)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    if not insps:
        st.info("The inspection queue is empty. Request an inspection from a satellite anomaly.")
        return
    st.subheader("Inspection queue")
    df = pd.DataFrame([{
        "Inspection ID": i["inspection_code"],
        "Origin": ("DEMO" if i["is_demo"] else "Satellite anomaly" if i["anomaly_id"] else "Business-requested"),
        "Coordinates": f"{i['latitude']:.5f}, {i['longitude']:.5f}",
        "Description": (i["description"] or "")[:80],
        "Priority": config.PRIORITY_LABELS.get(i["priority"], i["priority"]),
        "Status": i["inspection_status"]} for i in insps])
    table(df)
    codes = {i["inspection_code"]: i["id"] for i in insps}
    code = st.selectbox("Open inspection", list(codes))
    iid = codes[code]
    insp = db.get_inspection(iid)
    an = db.get_anomaly(insp["anomaly_id"]) if insp["anomaly_id"] else None
    st.markdown(f"### {code} · {insp['inspection_status']}")
    c1, c2 = st.columns([2, 3])
    with c1:
        if an and an.get("preview_path"):
            st.image(an["preview_path"], caption="Satellite preview" + (" (DEMO — synthetic)" if an["is_demo"] else ""))
        else:
            st.caption("No stored satellite preview for this point.")
        show_map(mi.point_map(insp["latitude"], insp["longitude"], code), f"pm_{iid}", 220)
    with c2:
        st.write(f"**Coordinates:** {insp['latitude']:.5f}, {insp['longitude']:.5f}")
        st.write(f"**Detection reason:** {(an or {}).get('reason') or insp['description']}")
        if an:
            st.write(f"**Anomaly:** {an['anomaly_code']} · {an['anomaly_type']} · "
                     f"{an['anomaly_area'] or 0:.2f} ha · observed {an['observation_date'] or 'n/a'}")
            side = max(30.0, (an["anomaly_area"] or 0) ** 0.5 * 100 * 1.5)
            st.write(f"**Estimated inspection area:** about {(side / 100) ** 2:.2f} ha "
                     f"(~{side:.0f} m × {side:.0f} m including margin)")
        st.write(f"**Priority:** {config.PRIORITY_LABELS.get(insp['priority'])}")
        if insp["is_demo"]:
            st.markdown(ui.banner("demo", "DEMO record — not a real observation."), unsafe_allow_html=True)

    key = f"sim_{iid}"
    if st.button("Simulate Drone Verification", type="primary", key=f"b_{iid}"):
        st.session_state[key] = True
    if not st.session_state.get(key):
        return
    st.markdown(ui.banner("sim", drone.SIM_LABEL), unsafe_allow_html=True)
    up = st.file_uploader("Upload a detailed aerial photograph (JPG/PNG/WEBP)",
                          type=None, key=f"up_{iid}")
    if up is not None:
        sig = f"{up.name}-{up.size}"
        if st.session_state.get(f"sig_{iid}") != sig:
            try:
                path = drone.validate_and_save_upload(up.name, up.getvalue())
                drone.attach_image(iid, path, False)
                st.session_state[f"sig_{iid}"] = sig
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Use built-in demonstration image", key=f"demo_{iid}"):
        drone.attach_image(iid, drone.ensure_demo_image(), True)
        st.rerun()
    insp = db.get_inspection(iid)
    img = insp.get("uploaded_image_path")
    if not img:
        return
    st.image(img, caption="Inspection image")
    if insp["image_is_demo"]:
        st.markdown(ui.banner("demo", drone.DEMO_IMAGE_NOTICE), unsafe_allow_html=True)
    if st.button("Analyse image and generate verification summary", key=f"an_{iid}"):
        with open(img, "rb") as fh:
            data = fh.read()
        mime = "image/png" if img.lower().endswith(".png") else "image/jpeg"
        with st.spinner("Analysing…"):
            text, err = gem.analyze_image(data, mime)
        if text:
            summary = ("AI-GENERATED IMAGE INTERPRETATION (hypotheses, unverified)\n\n" + text)
        else:
            if err:
                st.info(f"Gemini image analysis unavailable ({err}). Showing a rule-based summary.")
            summary = drone.deterministic_verification_summary(insp, an, bool(insp["image_is_demo"]))
        db.update_inspection(iid, ai_summary=summary, inspection_status="Analysed")
        st.rerun()
    if insp.get("ai_summary"):
        st.subheader("Verification summary")
        st.write(insp["ai_summary"])
        st.caption("Observations are not proof of chemical pollution, toxicity or the cause "
                   "of a change. Field measurements are required.")


# -------------------------------------------------------- reports & finance
def page_reports():
    st.title("Reports & Finance")
    st.subheader("Financial analytics (illustrative)")
    st.markdown(ui.banner("info", fin.ILLUSTRATIVE_NOTE), unsafe_allow_html=True)
    st.caption("The default values below are placeholders only. Replace them with your own "
               "figures or quotations.")
    kpis = db.get_kpis()
    c1, c2, c3 = st.columns(3)
    ha = c1.number_input("Monitored area (ha)", min_value=0.0, value=float(round(kpis["total_area_ha"])), step=10.0)
    n = c2.number_input("Planned inspections per year", min_value=0, value=10, step=1)
    conv = c3.number_input("Conventional cost per inspection", min_value=0.0, value=2000.0, step=100.0)
    c4, c5, _ = st.columns(3)
    targ = c4.number_input("Targeted cost per inspection", min_value=0.0, value=800.0, step=100.0)
    sub = c5.number_input("TerraSignal monthly subscription", min_value=0.0, value=300.0, step=50.0)
    try:
        r = fin.calculate(conv, targ, int(n), sub)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["fin_res"] = r
    m = st.columns(4)
    m[0].metric("Conventional cost / inspection", f"{r.conventional_cost:,.2f}")
    m[1].metric("Targeted cost / inspection", f"{r.targeted_cost:,.2f}")
    m[2].metric("Est. annual savings", f"{r.annual_gross_savings:,.2f}")
    m[3].metric("Annual platform cost", f"{r.annual_platform_cost:,.2f}")
    m = st.columns(3)
    m[0].metric("Est. annual net benefit", f"{r.annual_net_benefit:,.2f}")
    m[1].metric("ROI", "n/a (no platform cost)" if r.roi_percent is None else f"{r.roi_percent:,.1f}%")
    m[2].metric("Break-even inspections / year",
                "not reached" if r.break_even_inspections is None else str(r.break_even_inspections))
    if r.savings_per_inspection <= 0:
        st.warning("Targeted inspections cost as much as or more than conventional ones in "
                   "this scenario, so there are no inspection savings.")
    import plotly.graph_objects as go
    pts = fin.cumulative_curve(r)
    fig = go.Figure(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                               line=dict(color=config.COLORS["primary"], width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color=config.COLORS["muted"])
    fig.update_layout(height=280, xaxis_title="Inspections per year", yaxis_title="Net benefit",
                      margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    chart(fig)
    areas = db.list_areas()
    labels = {None: "— no specific area —", **{a["id"]: a["name"] for a in areas}}
    sc_area = st.selectbox("Link scenario to area (optional)", list(labels), format_func=labels.get)
    if st.button("Save scenario"):
        db.add_financial_scenario(sc_area, conv, targ, int(n), sub, r.annual_net_benefit)
        st.success("Scenario saved. It now appears in the Overview savings KPI.")

    st.markdown("---")
    st.subheader("Inspection report")
    insps = db.list_inspections()
    if not insps:
        st.info("Create an inspection first (Satellite Scan → Request Detailed Inspection).")
        return
    codes = {i["inspection_code"]: i["id"] for i in insps}
    code = st.selectbox("Inspection", list(codes), key="rep_insp")
    iid = codes[code]
    insp = db.get_inspection(iid)
    an = db.get_anomaly(insp["anomaly_id"]) if insp["anomaly_id"] else None
    area = db.get_area(insp["monitoring_area_id"]) if insp["monitoring_area_id"] else None
    ai_img = insp.get("ai_summary")
    if ai_img and ai_img.startswith("VERIFICATION SUMMARY (rule-based"):
        ai_img = None
    gem_ok, _ = gem.is_available()
    use_gem = st.checkbox("Use Gemini to write the report text", value=gem_ok, disabled=not gem_ok)
    skey = f"rep_{iid}"
    if st.button("Generate report text"):
        ctx = rg.build_context(insp, an, area, r, ai_img)
        text, src = None, "deterministic generator"
        if use_gem:
            with st.spinner("Generating with Gemini…"):
                text, err = gem.generate_report(ctx)
            if text:
                src = "Gemini (AI-generated hypotheses are labelled)"
            else:
                st.info(f"Gemini unavailable ({err}); using the deterministic generator.")
        if not text:
            text = rg.deterministic_report(insp, an, area, ai_img)
        st.session_state[skey] = (text, src)
    if skey in st.session_state:
        text, src = st.session_state[skey]
        st.caption(f"Report source: {src}")
        st.write(text)
        pdf = rg.build_pdf(insp, an, area, text, ai_img, r, src)
        st.download_button("Download PDF Report", pdf, file_name=f"terrasignal_{code}.pdf",
                           mime="application/pdf", type="primary",
                           on_click=lambda: db.update_inspection(iid, inspection_status="Report generated"))
    st.caption(config.DISCLAIMER)


# --------------------------------------------------------------------- main
page = sidebar()
{"Overview": page_overview, "Satellite Scan": page_scan, "Monitoring Areas": page_areas,
 "Drone Verification": page_drone, "Reports & Finance": page_reports}[page]()
