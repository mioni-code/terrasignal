"""Design system: CSS and small HTML components for the Streamlit UI."""
from __future__ import annotations

from html import escape

import config

C = config.COLORS

CSS = f"""
<style>
.stApp {{ background: {C['bg']}; color: {C['text']}; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {C['border']}; }}
h1, h2, h3 {{ color: {C['primary']}; letter-spacing: -0.01em; }}
.ts-brand {{ font-size: 1.5rem; font-weight: 700; color: {C['primary']}; margin-bottom: 0; }}
.ts-tag {{ font-size: 0.78rem; color: {C['muted']}; margin-bottom: 1rem; }}
.ts-card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 14px;
            padding: 1rem 1.1rem; margin-bottom: 0.8rem; }}
.ts-kpi-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; color: {C['muted']}; }}
.ts-kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {C['primary']}; line-height: 1.2; }}
.ts-kpi-sub {{ font-size: 0.75rem; color: {C['muted']}; }}
.ts-banner {{ border-radius: 12px; padding: .7rem 1rem; margin: .4rem 0 .8rem 0; font-size: .9rem; border: 1px solid; }}
.ts-demo {{ background: #FFF4E5; border-color: #E8B25C; color: #7A4B00; }}
.ts-sim  {{ background: #FDECEA; border-color: #E4A09B; color: #8C1D18; font-weight: 700; letter-spacing: .03em; }}
.ts-info {{ background: #EEF4EE; border-color: {C['secondary']}; color: {C['primary']}; }}
.ts-badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; color: #fff; }}
.ts-status {{ font-size: .78rem; color: {C['muted']}; line-height: 1.8; }}
.ts-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
    background: {C['primary']}; border: none; border-radius: 10px; }}
div.stButton > button {{ border-radius: 10px; }}
@media (max-width: 700px) {{ .ts-kpi-value {{ font-size: 1.3rem; }} }}
</style>
"""


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f"<div class='ts-card'><div class='ts-kpi-label'>{escape(label)}</div>"
            f"<div class='ts-kpi-value'>{escape(value)}</div>"
            f"<div class='ts-kpi-sub'>{escape(sub)}</div></div>")


def banner(kind: str, text: str) -> str:
    return f"<div class='ts-banner ts-{kind}'>{escape(text)}</div>"


def badge(priority: str, demo: bool = False) -> str:
    label = config.PRIORITY_LABELS.get(priority, priority) + (" · DEMO" if demo else "")
    return f"<span class='ts-badge' style='background:{C.get(priority, C['muted'])}'>{escape(label)}</span>"


def status_dot(ok: bool, label: str, detail: str) -> str:
    color = C["Normal"] if ok else C["Attention"]
    return (f"<div><span class='ts-dot' style='background:{color}'></span>"
            f"<b>{escape(label)}</b>: {escape(detail)}</div>")
