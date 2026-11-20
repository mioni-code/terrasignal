# TerraSignal — Macro-to-Micro Environmental Monitoring (MVP)

*Detect Early. Verify Precisely. Protect Nature.*

Prototype for a startup demonstration. **Not** a production or official environmental
monitoring system.

Workflow: satellite image → change detection → map alert → inspection request →
simulated drone verification → environmental report → financial analysis.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                    # optional: add GEMINI_API_KEY
streamlit run app.py
```

Run the offline tests: `python tests/test_core.py && python tests/test_fixes.py`

## What is real, what is simulated

| Component | Status |
|---|---|
| Sentinel-2 L2A search + windowed read (Planetary Computer STAC) | Real (needs internet) |
| NDVI / NDWI, SCL cloud masking, two-period change detection | Real computation on real data |
| Priority score | Heuristic screening indicator (not an AI confidence) |
| Drone verification | **Simulated** — no flight, no drone control |
| Built-in demonstration image | **Synthetic, illustrative only** |
| Seeded records (`[DEMO]`) and DEMO MODE scenario | **Synthetic**, always labelled, never counted as real |
| Gemini image/report analysis | Optional; app works without it |
| Financial figures | Illustrative, from user-entered assumptions only |

## Project layout

```
app.py                      Streamlit UI (5 pages)
config.py  database.py      settings, SQLite layer
modules/satellite_data.py   STAC search, windowed raster reads, demo arrays
modules/vegetation_analysis.py  NDVI, SCL mask, median composite
modules/water_analysis.py   NDWI, water extent change
modules/anomaly_detection.py    thresholding, connected regions, priority, orchestration
modules/map_interface.py    Folium maps, overlays, legend
modules/monitoring_areas.py business-requested areas
modules/drone_simulation.py upload validation, demo image, statuses
modules/gemini_service.py   optional google-genai calls
modules/financial_analytics.py  savings / ROI / break-even
modules/report_generator.py deterministic report + PDF (reportlab)
tests/test_core.py          offline tests (finance, indices, DB, PDF)
tests/test_fixes.py         offline tests (polygon mask, map args, NDWI terminology, report wording)
```

## Method notes

- Reflectance = DN × 1e-4 with the STAC `raster:bands` offset (or −1000 for processing
  baseline ≥ 04.00). Valid pixels: SCL 4, 5, 6.
- Each period is a per-pixel median of up to 4 scenes from one MGRS tile (so both periods
  share a pixel grid). Change = recent − baseline.
- Regions = 8-connected pixels beyond the threshold; regions under the minimum size are dropped.
- Default baseline is the same season one year earlier to limit seasonal false alarms.
- Index changes do not identify causes; NDWI does not detect pollution.
- NDVI and NDWI values are remotely sensed indicators, not field-confirmed measurements.

## Polygon masking

The bounding box is used only to decide which window of each Sentinel-2 scene is read.
The drawn / saved GeoJSON polygon is then projected from EPSG:4326 into the raster's own
CRS, converted to pixel space with the inverse of the raster transform, and rasterised
(pixel centres inside the polygon). Pixels outside it are set to NaN in every per-scene
index before compositing, so they are absent from both composites, the change map and
anomaly detection. The share of the bounding rectangle actually analysed is shown in the UI.

## Water terminology

Two quantities are reported separately:
- **NDWI index change** (recent − baseline): a reflectance-index difference. Regions are typed
  "NDWI decrease/increase (index change)" and are not called water-surface changes.
- **Estimated open-water extent change**: hectares where NDWI > 0 in one period but not the
  other (scene-wide summary, and per region in the detection reason).

## Changelog (review fixes)

1. Saved-area map: `draw_map(center=..., zoom=..., highlight_geom=...)` with keywords.
2. Real polygon masking (see above), also applied in DEMO mode.
3. NDWI index change vs. estimated open-water extent are separated and labelled accurately.
4. "CONFIRMED MEASUREMENTS" replaced by "SATELLITE-DERIVED INDEX MEASUREMENTS" in
   deterministic reports, PDF, and the Gemini report prompt, with an explicit
   not-field-confirmed statement.
5. New tests in `tests/test_fixes.py`.

## Known limitations

- AOIs above 50 km² are rejected; both periods must have scenes on one common tile.
- Anomaly outlines are region bounding boxes, not traced contours.
- Polygon masking uses pixel-centre inclusion; edge pixels partly inside the polygon may be
  excluded. Very small polygons (under one 10 m pixel) are rejected with an error.
- Polygon masking and the full real-data pipeline were tested only with synthetic arrays and
  a stand-in CRS/transform, not with real Sentinel-2 scenes (rasterio and network access were
  unavailable in the development environment).
- Seeded demo record type text and any previously saved anomalies in an existing database
  keep their old labels; delete `data/terrasignal.db` to re-seed.
- Monitoring "frequency" is a stored preference; no scheduler runs scans automatically.
- The Gemini model name is configurable (`GEMINI_MODEL`); check it against current availability.
