"""Deterministic report text, Gemini context builder and PDF generation."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import config
from modules import drone_simulation as drone
from modules import financial_analytics as fin


def is_demo_record(insp: dict, anomaly: dict | None) -> bool:
    return bool(insp.get("is_demo") or (anomaly and anomaly.get("is_demo")))


def build_context(insp: dict, anomaly: dict | None, area: dict | None,
                  fin_res: fin.FinancialResult | None, ai_image_summary: str | None) -> str:
    """Plain-text data block (used for Gemini and the deterministic report)."""
    L = [f"Inspection ID: {insp.get('inspection_code')}",
         f"Record type: {'DEMO (not a satellite observation)' if is_demo_record(insp, anomaly) else 'Real satellite-derived record'}",
         f"Monitoring area: {area['name'] if area else 'n/a'}",
         f"Coordinates: {insp.get('latitude'):.5f}, {insp.get('longitude'):.5f}"]
    if anomaly:
        L += [f"Anomaly ID: {anomaly['anomaly_code']}",
              f"Anomaly type: {anomaly['anomaly_type']}",
              f"Anomaly area (ha): {anomaly.get('anomaly_area')}",
              f"NDVI change (mean, satellite-derived index): {anomaly.get('ndvi_change')}",
              f"NDWI change (mean, satellite-derived index): {anomaly.get('ndwi_change')}",
              f"Recent observation date: {anomaly.get('observation_date')}",
              f"Baseline date: {anomaly.get('baseline_date')}",
              f"Cloud cover of scene (%): {anomaly.get('cloud_cover')}",
              f"Heuristic priority: {config.PRIORITY_LABELS.get(anomaly['priority'])} "
              f"(score {anomaly.get('priority_score')}, a screening indicator)",
              f"Detection reason: {anomaly.get('reason')}",
              f"Data source: {anomaly['source']}"]
    L.append(config.INDEX_DISCLAIMER)
    L.append(f"Inspection status: {insp['inspection_status']}")
    if ai_image_summary:
        L.append("Inspection image analysis (AI-generated, unverified):\n" + ai_image_summary)
    if fin_res:
        L.append(f"Illustrative annual net benefit: {fin_res.annual_net_benefit:,.2f} "
                 "(user assumptions only)")
    return "\n".join(L)


def deterministic_report(insp: dict, anomaly: dict | None, area: dict | None,
                         ai_image_summary: str | None) -> str:
    demo = is_demo_record(insp, anomaly)
    out = ["1. EXECUTIVE SUMMARY"]
    if anomaly:
        out.append(f"A {anomaly['anomaly_type'].lower()} of approx. "
                   f"{anomaly.get('anomaly_area') or 0:.2f} ha was flagged near "
                   f"{insp['latitude']:.5f}, {insp['longitude']:.5f}. Priority: "
                   f"{config.PRIORITY_LABELS.get(anomaly['priority'])} (heuristic screening "
                   "indicator).")
    else:
        out.append("A business-requested inspection point was created without a satellite "
                   f"anomaly at {insp['latitude']:.5f}, {insp['longitude']:.5f}.")
    if demo:
        out.append("NOTE: this is a DEMO record. It is not a real observation.")
    out += ["", "2. DETECTED CHANGES (SATELLITE-DERIVED INDEX MEASUREMENTS)",
            config.INDEX_DISCLAIMER]
    if anomaly:
        if anomaly.get("ndvi_change") is not None:
            out.append(f"- Mean NDVI change: {anomaly['ndvi_change']:+.3f}")
        if anomaly.get("ndwi_change") is not None:
            out.append(f"- Mean NDWI change: {anomaly['ndwi_change']:+.3f}")
        out.append(f"- Area: {anomaly.get('anomaly_area') or 0:.2f} ha")
        out.append(f"- Observation date: {anomaly.get('observation_date') or 'n/a'}; "
                   f"baseline: {anomaly.get('baseline_date') or 'n/a'}")
    else:
        out.append("- No satellite measurements are attached to this inspection.")
    out += ["", "3. POSSIBLE ENVIRONMENTAL SIGNIFICANCE (HYPOTHESES)",
            "- A reduction in vegetation index can reflect disturbance, but also "
            "harvest, seasonal senescence, drought, tillage or residual cloud.",
            "- An NDWI change reflects a change in a reflectance-based water/wetness index. It does "
            "not by itself confirm a change in open-water extent (see the separate NDWI > 0 "
            "water-mask estimate in the detection reason) and is not a measure of water quality.",
            "", "4. EVIDENCE AVAILABLE",
            f"- Satellite: {anomaly['source'] if anomaly else 'none'}",
            f"- Drone verification: {insp['inspection_status']} (SIMULATED; no real flight)",
            "- Aerial image: " + ("illustrative demo image" if insp.get("image_is_demo")
                                  else "uploaded by user" if insp.get("uploaded_image_path")
                                  else "none"),
            "- AI image interpretation: " + ("available (unverified)" if ai_image_summary
                                             else "not available"),
            "", "5. UNCERTAINTIES",
            "- Index changes do not identify the cause of the change.",
            "- Single-date composites and cloud masking can leave artefacts.",
            "- Priority score is a heuristic, not a probability.",
            "", "6. RECOMMENDED NEXT STEPS",
            "- Review the area in a newer satellite image.",
            "- If still unexplained, arrange a qualified on-site inspection.",
            "", "7. ADDITIONAL DATA REQUIRED",
            "- Field observations and, where relevant, laboratory measurements.",
            "- Land-management records (harvest, construction, permitted activities)."]
    return "\n".join(out)


# --------------------------------------------------------------------- PDF
def build_pdf(insp: dict, anomaly: dict | None, area: dict | None,
              report_text: str, ai_image_summary: str | None,
              fin_res: fin.FinancialResult | None, report_source: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    green = colors.HexColor(config.COLORS["primary"])
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=green, fontSize=12, spaceBefore=8)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    P = lambda t, st=body: Paragraph(escape(str(t)).replace("\n", "<br/>"), st)  # noqa: E731

    demo = is_demo_record(insp, anomaly)
    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 9 * mm, "TerraSignal — Preliminary environmental "
                          "screening report. Not an official assessment.")
        canvas.drawRightString(195 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=16 * mm,
                            title=f"TerraSignal {insp.get('inspection_code')}")
    s = []
    head = Table([[Paragraph("<font color='white'><b>TerraSignal</b></font>",
                             ParagraphStyle("t", fontSize=18, leading=22)),
                   Paragraph("<font color='white'>Preliminary Environmental Screening Report</font>",
                             ParagraphStyle("t2", fontSize=10, alignment=2))]],
                 colWidths=[70 * mm, 110 * mm])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), green),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("TOPPADDING", (0, 0), (-1, -1), 10),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    s += [head, Spacer(1, 4), P(config.TAGLINE, small), Spacer(1, 6)]
    if demo:
        s += [P("DEMO RECORD — contains illustrative data. NOT a real satellite observation.",
                ParagraphStyle("d", parent=body, textColor=colors.HexColor("#B3261E"),
                               fontName="Helvetica-Bold")), Spacer(1, 4)]

    rows = [["Inspection ID", insp.get("inspection_code")],
            ["Monitoring area", area["name"] if area else "n/a"],
            ["Coordinates", f"{insp['latitude']:.5f}, {insp['longitude']:.5f}"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Drone verification status",
             f"{insp['inspection_status']} (SIMULATED — no real flight)"]]
    if anomaly:
        rows += [["Anomaly", f"{anomaly['anomaly_code']} — {anomaly['anomaly_type']}"],
                 ["Anomaly area", f"{anomaly.get('anomaly_area') or 0:.2f} ha"],
                 ["Satellite data source", anomaly["source"]],
                 ["Recent image date", anomaly.get("observation_date") or "n/a"],
                 ["Baseline date", anomaly.get("baseline_date") or "n/a"],
                 ["Cloud cover", f"{anomaly['cloud_cover']:.1f}%"
                  if anomaly.get("cloud_cover") is not None else "n/a"],
                 ["NDVI change", f"{anomaly['ndvi_change']:+.3f}"
                  if anomaly.get("ndvi_change") is not None else "n/a"],
                 ["NDWI change", f"{anomaly['ndwi_change']:+.3f}"
                  if anomaly.get("ndwi_change") is not None else "n/a"],
                 ["Priority (heuristic)",
                  f"{config.PRIORITY_LABELS.get(anomaly['priority'])} — preliminary "
                  "screening indicator"]]
    t = Table([[P(k, small), P(v)] for k, v in rows], colWidths=[48 * mm, 132 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(config.COLORS["border"])),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(config.COLORS["bg"])),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [t, Spacer(1, 6)]

    s += [P("Detection methodology", h2),
          P("Sentinel-2 Level-2A surface reflectance (Microsoft Planetary Computer). "
            "NDVI = (B08−B04)/(B08+B04); NDWI = (B03−B08)/(B03+B08). Cloud, shadow and "
            "invalid pixels are masked with the Scene Classification Layer. A baseline and "
            "a recent period are median-composited; pixels whose index change exceeds a "
            "user threshold are grouped into regions and small regions are discarded. "
            "Priority is a heuristic screening indicator, not an AI confidence. "
            + config.INDEX_DISCLAIMER), ]

    img_path = insp.get("uploaded_image_path")
    if img_path and Path(img_path).exists():
        s.append(P("Inspection image", h2))
        if insp.get("image_is_demo"):
            s.append(P(drone.DEMO_IMAGE_NOTICE, small))
        try:
            s.append(RLImage(img_path, width=120 * mm, height=80 * mm, kind="proportional"))
        except Exception:
            s.append(P("(Image could not be embedded.)", small))
    if ai_image_summary:
        s += [P("AI interpretation of inspection image (unverified hypotheses)", h2),
              P(ai_image_summary)]

    s += [P(f"Report text ({report_source})", h2), P(report_text)]

    if fin_res:
        s += [P("Financial estimates (illustrative)", h2),
              P(f"Savings per inspection: {fin_res.savings_per_inspection:,.2f}\n"
                f"Annual gross savings: {fin_res.annual_gross_savings:,.2f}\n"
                f"Annual platform cost: {fin_res.annual_platform_cost:,.2f}\n"
                f"Estimated annual net benefit: {fin_res.annual_net_benefit:,.2f}\n"
                f"ROI: {'n/a' if fin_res.roi_percent is None else f'{fin_res.roi_percent:,.1f}%'}\n"
                f"Break-even inspections: "
                f"{'never at these inputs' if fin_res.break_even_inspections is None else fin_res.break_even_inspections}"),
              P(fin.ILLUSTRATIVE_NOTE, small)]

    s += [P("Limitations and uncertainty", h2), P(config.DISCLAIMER, small)]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
