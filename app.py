"""MoleculeIQ Agent — Gradio Space entry point."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Generator

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Export HF_TOKEN before starting the app."
    )


def _prewarm_models() -> None:
    for name, loader in [
        ("toxicity predictor", "agents.toxicity_checker._load_model"),
        ("odor predictor",     "agents.odor_predictor._load_model"),
    ]:
        try:
            mod, fn = loader.rsplit(".", 1)
            import importlib
            load_fn = getattr(importlib.import_module(mod), fn)
            logger.info("Pre-warming %s…", name)
            load_fn()
        except Exception as exc:
            logger.warning("%s pre-warm failed: %s", name, exc)


# ── Custom Gradio Theme ─────────────────────────────────────────────────

class MoleculeIQTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.indigo,
            secondary_hue=colors.slate,
            neutral_hue=colors.slate,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_lg,
            font=[
                fonts.GoogleFont("Inter"),
                fonts.Font("system-ui"),
                fonts.Font("sans-serif"),
            ],
            font_mono=[
                fonts.GoogleFont("JetBrains Mono"),
                fonts.Font("ui-monospace"),
            ],
        )
        self.set(
            # ── Backgrounds
            body_background_fill="#f1f5f9",
            background_fill_primary="#ffffff",
            background_fill_secondary="#f8fafc",

            # ── Body text
            body_text_color="#1e293b",
            body_text_color_subdued="#64748b",
            body_text_size="14px",
            body_text_weight="400",

            # ── Borders
            border_color_primary="#e2e8f0",
            border_color_accent="#6366f1",
            border_color_accent_subdued="#c7d2fe",

            # ── Blocks / panels
            block_background_fill="#ffffff",
            block_border_color="#e2e8f0",
            block_border_width="1px",
            block_label_background_fill="#f8fafc",
            block_label_text_color="#64748b",
            block_label_text_size="*text_sm",
            block_label_text_weight="600",
            block_shadow="none",
            block_radius="12px",
            block_padding="20px",
            block_title_text_color="#1e293b",
            block_title_text_size="*text_sm",
            block_title_text_weight="700",
            block_title_background_fill="#f8fafc",
            block_title_border_color="#e2e8f0",
            block_title_border_width="0px",

            # ── Input fields
            input_background_fill="#f8fafc",
            input_background_fill_focus="#ffffff",
            input_background_fill_hover="#ffffff",
            input_border_color="#e2e8f0",
            input_border_color_focus="#6366f1",
            input_border_color_hover="#cbd5e1",
            input_border_width="1.5px",
            input_shadow="none",
            input_shadow_focus="0 0 0 3px rgba(99,102,241,0.12)",
            input_radius="10px",
            input_padding="12px 14px",
            input_placeholder_color="#94a3b8",
            input_text_weight="400",

            # ── Primary button (Analyse)
            button_primary_background_fill="#1e293b",
            button_primary_background_fill_hover="#0f172a",
            button_primary_text_color="#ffffff",
            button_primary_text_color_hover="#ffffff",
            button_primary_border_color="#1e293b",
            button_primary_border_color_hover="#0f172a",
            button_primary_shadow="none",
            button_primary_shadow_hover="none",
            button_primary_shadow_active="none",

            # ── Secondary button (chips)
            button_secondary_background_fill="#f1f5f9",
            button_secondary_background_fill_hover="#e2e8f0",
            button_secondary_text_color="#475569",
            button_secondary_text_color_hover="#1e293b",
            button_secondary_border_color="#e2e8f0",
            button_secondary_border_color_hover="#cbd5e1",
            button_secondary_shadow="none",
            button_secondary_shadow_hover="none",
            button_secondary_shadow_active="none",

            # ── Button sizing
            button_large_padding="11px 26px",
            button_large_radius="10px",
            button_large_text_size="14px",
            button_large_text_weight="700",
            button_medium_padding="9px 20px",
            button_medium_radius="10px",
            button_medium_text_size="13px",
            button_medium_text_weight="600",
            button_small_padding="5px 14px",
            button_small_radius="100px",
            button_small_text_size="12px",
            button_small_text_weight="500",
            button_border_width="1px",
            button_transform_hover="none",
            button_transition="background 0.15s ease",

            # ── Panels
            panel_background_fill="#f8fafc",
            panel_border_color="#e2e8f0",
            panel_border_width="1px",

            # ── Tables
            table_even_background_fill="#f8fafc",
            table_odd_background_fill="#ffffff",
            table_border_color="#e2e8f0",
            table_radius="8px",
            table_row_focus="#eef2ff",

            # ── Prose (Markdown)
            prose_text_size="14.5px",
            prose_text_weight="400",
            prose_header_text_weight="700",

            # ── Section headers
            section_header_text_size="11px",
            section_header_text_weight="800",

            # ── Shadows off
            shadow_drop="none",
            shadow_drop_lg="none",
            shadow_spread="0px",

            # ── Links
            link_text_color="#6366f1",
            link_text_color_hover="#4f46e5",
            link_text_color_visited="#6366f1",
            link_text_color_active="#4f46e5",

            # ── Misc
            color_accent="#6366f1",
            color_accent_soft="#eef2ff",
            checkbox_background_color="#f8fafc",
            checkbox_border_color="#cbd5e1",
            layout_gap="0px",
            form_gap_width="0px",
        )


# ── UI helpers ──────────────────────────────────────────────────────────

STEP_MAP = [
    ("researching",         "🔍", "Database lookup"),
    ("molecule identified", "🧬", "Compound identified"),
    ("running safety",      "🛡", "Safety review"),
    ("safety analysis",     "✓",  "Safety review"),
    ("running toxicity",    "⚗",  "Toxicity screening"),
    ("toxicity check",      "⚗",  "Toxicity screening"),
    ("predicting odor",     "〜", "Odor profile"),
    ("odor profile",        "〜", "Odor profile"),
    ("cross-checking",      "↻",  "Cross-checking sources"),
    ("review verdict",      "↻",  "Cross-checking sources"),
    ("formatting",          "📋", "Building report"),
    ("done",                "✓",  "Complete"),
    ("failed",              "✕",  "Failed"),
    ("skipping",            "→",  "Skipped"),
    ("no smiles",           "→",  "Skipped"),
]


def _step_meta(msg: str) -> tuple[str, str]:
    lower = msg.lower()
    for key, icon, label in STEP_MAP:
        if key in lower:
            return icon, label
    return "·", msg[:36]


def _progress_html(lines: list[str]) -> str:
    FF = "font-family:Inter,system-ui,sans-serif;"
    if not lines:
        return (
            f'<div style="{FF}padding:32px 8px;text-align:center;'
            f'color:#94a3b8;font-size:13px;font-style:italic;">'
            f'Waiting for a query…</div>'
        )

    # Deduplicate — keep last occurrence per label
    deduped: list[tuple[str, str, str]] = []
    seen: dict[str, int] = {}
    for line in lines:
        icon, label = _step_meta(line)
        sub = re.sub(r"(?i)^" + re.escape(label) + r"\s*:?\s*", "", line).strip() or line[:52]
        entry = (icon, label, sub[:55])
        if label in seen:
            deduped[seen[label]] = entry
        else:
            seen[label] = len(deduped)
            deduped.append(entry)

    rows = []
    last_idx = len(deduped) - 1

    for j, (icon, label, sub) in enumerate(deduped):
        is_last = j == last_idx
        is_terminal = any(k in label.lower() for k in ("complete", "failed", "skipped"))
        is_active = is_last and not is_terminal

        if is_active:
            dot_bg, dot_bd, dot_fg = "#eef2ff", "#a5b4fc", "#4f46e5"
            lbl_color = "#4f46e5"
        elif "failed" in label.lower():
            dot_bg, dot_bd, dot_fg = "#fef2f2", "#fca5a5", "#dc2626"
            lbl_color = "#dc2626"
        else:
            dot_bg, dot_bd, dot_fg = "#f0fdf4", "#86efac", "#16a34a"
            lbl_color = "#1e293b"

        line_div = (
            f'<div style="width:2px;min-height:18px;'
            f'background:{"#c7d2fe" if is_active else "#bbf7d0"};'
            f'margin:2px auto;"></div>'
            if not is_last else ""
        )

        rows.append(
            f'<div style="{FF}display:flex;gap:12px;align-items:flex-start;">'
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'            width:32px;flex-shrink:0;">'
            f'  <div style="width:32px;height:32px;border-radius:50%;'
            f'              background:{dot_bg};border:1.5px solid {dot_bd};'
            f'              display:flex;align-items:center;justify-content:center;'
            f'              font-size:14px;color:{dot_fg};">{icon}</div>'
            f'  {line_div}'
            f'</div>'
            f'<div style="padding-top:4px;min-width:0;flex:1;'
            f'            padding-bottom:{"0" if is_last else "6px"};">'
            f'  <div style="font-size:13.5px;font-weight:600;color:{lbl_color};'
            f'              line-height:1.3;">{label}</div>'
            f'  <div style="font-size:11.5px;color:#64748b;margin-top:2px;'
            f'              font-family:JetBrains Mono,ui-monospace,monospace;'
            f'              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'              max-width:185px;">{sub}</div>'
            f'</div>'
            f'</div>'
        )

    return f'<div style="display:flex;flex-direction:column;">{"".join(rows)}</div>'


def _verdict_html(badge: str, compound: str, formula: str) -> str:
    FF = "font-family:Inter,system-ui,sans-serif;"
    if not badge:
        return (
            f'<div style="{FF}display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;gap:10px;padding:44px 20px;border-radius:14px;'
            f'background:#f8fafc;border:1.5px dashed #cbd5e1;color:#94a3b8;">'
            f'<div style="font-size:32px;">🔬</div>'
            f'<div style="font-size:13.5px;font-style:italic;">Safety verdict will appear here</div>'
            f'</div>'
        )

    badge_l = badge.lower()
    if "generally safe" in badge_l and "consult" not in badge_l:
        bg, border, color = "#f0fdf4", "#86efac", "#14532d"
        icon_bg, icon, kicker = "#dcfce7", "✅", "Safe to use"
    elif "consult" in badge_l or ("caution" in badge_l and "avoid" not in badge_l):
        bg, border, color = "#fffbeb", "#fde68a", "#78350f"
        icon_bg, icon, kicker = "#fef9c3", "⚠️", "Use with caution"
    else:
        bg, border, color = "#fff1f2", "#fecaca", "#881337"
        icon_bg, icon, kicker = "#ffe4e6", "🚫", "Caution — significant risk"

    meta = ""
    if compound:
        meta += f'<span style="font-size:13px;font-weight:500;opacity:.8;">{compound}</span>'
    if formula:
        meta += (
            f'<span style="font-size:12px;'
            f'font-family:JetBrains Mono,ui-monospace,monospace;'
            f'background:rgba(255,255,255,.6);border:1px solid rgba(0,0,0,.08);'
            f'padding:2px 8px;border-radius:6px;margin-left:6px;">{formula}</span>'
        )

    return (
        f'<div style="{FF}display:flex;align-items:center;gap:18px;'
        f'padding:20px 22px;border-radius:14px;'
        f'border:1.5px solid {border};background:{bg};color:{color};">'
        f'<div style="width:52px;height:52px;border-radius:50%;background:{icon_bg};'
        f'            display:flex;align-items:center;justify-content:center;'
        f'            font-size:26px;flex-shrink:0;">{icon}</div>'
        f'<div style="min-width:0;">'
        f'  <div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
        f'              text-transform:uppercase;opacity:.6;margin-bottom:4px;">{kicker}</div>'
        f'  <div style="font-size:18px;font-weight:800;letter-spacing:-.025em;'
        f'              line-height:1.25;">{badge}</div>'
        f'  <div style="display:flex;align-items:center;flex-wrap:wrap;'
        f'              margin-top:7px;gap:6px;">{meta}</div>'
        f'</div>'
        f'</div>'
    )


EXAMPLE_QUESTIONS = [
    "Is aspartame safe daily?",
    "Is bleach dangerous?",
    "Is BPA harmful?",
    "Is alcohol safe daily?",
    "Is ibuprofen safe long-term?",
    "Is lead dangerous?",
]

# Surgical CSS — only for layout/spacing Gradio's theme tokens can't reach
CSS = """
/* ── Container reset ──────────────────── */
.gradio-container { max-width:100% !important; padding:0 !important; }
.gradio-container > .main { padding:0 !important; gap:0 !important; }
.gradio-container > .main > .wrap { padding:0 !important; gap:0 !important; }
footer { display:none !important; }

/* ── Nav ──────────────────────────────── */
#miq-nav { border-bottom:1px solid #e2e8f0 !important; }
#miq-nav > .block { padding:0 !important; background:#fff !important;
                    border:none !important; box-shadow:none !important; border-radius:0 !important; }

/* ── Search row ───────────────────────── */
#miq-search-row { background:#fff !important; border-bottom:1px solid #e2e8f0 !important;
                  padding:14px 24px !important; gap:10px !important; align-items:center !important; }
#miq-search-row > div { gap:10px !important; align-items:center !important; }
#miq-q { flex:1 !important; }
#miq-q > .wrap { border-radius:10px !important; }
#miq-q textarea, #miq-q input { font-size:15px !important; line-height:1.5 !important; }

/* Analyse button */
#miq-btn > .wrap { border-radius:10px !important; overflow:hidden !important; }
#miq-btn button {
    background:#1e293b !important; color:#fff !important;
    border:none !important; border-radius:10px !important;
    font-size:14px !important; font-weight:700 !important;
    padding:12px 28px !important; min-height:46px !important;
    letter-spacing:-.01em !important; box-shadow:none !important;
    transition:background .15s !important;
}
#miq-btn button:hover { background:#0f172a !important; }

/* ── Chips ────────────────────────────── */
#miq-chips { background:#fff !important; border-bottom:1px solid #e2e8f0 !important;
             padding:8px 24px !important; gap:6px !important; flex-wrap:wrap !important; }
#miq-chips > div { gap:6px !important; flex-wrap:wrap !important;
                   background:transparent !important; border:none !important; }
#miq-chips button {
    border-radius:100px !important; font-size:12px !important; font-weight:500 !important;
    padding:5px 14px !important; height:auto !important;
    background:#f1f5f9 !important; border:1px solid #e2e8f0 !important;
    color:#475569 !important; box-shadow:none !important; transition:all .14s !important;
}
#miq-chips button:hover {
    background:#eef2ff !important; border-color:#a5b4fc !important; color:#4f46e5 !important;
}

/* ── Body grid ────────────────────────── */
#miq-body { gap:0 !important; background:#f1f5f9 !important;
            align-items:stretch !important; }
#miq-body > div { gap:0 !important; }

/* ── Pipeline column ──────────────────── */
#miq-pipeline {
    background:#fff !important; border-right:1px solid #e2e8f0 !important;
    border-radius:0 !important; border-top:none !important;
    border-bottom:none !important; border-left:none !important;
    box-shadow:none !important; padding:20px 18px !important; min-height:500px !important;
}
#miq-pipeline > div { gap:0 !important; }
#miq-pipeline .block {
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
}

/* ── Results column ───────────────────── */
#miq-results {
    padding:18px 22px !important; gap:14px !important;
    background:#f1f5f9 !important; border-radius:0 !important;
    border:none !important; box-shadow:none !important;
}
#miq-results > div { gap:14px !important; }
#miq-results .block {
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
}

/* ── Verdict block ────────────────────── */
#miq-verdict .block {
    padding:0 !important; background:transparent !important;
    border:none !important; box-shadow:none !important;
}

/* ── Report card ──────────────────────── */
#miq-report {
    background:#fff !important; border:1px solid #e2e8f0 !important;
    border-radius:14px !important; overflow:hidden !important;
    box-shadow:none !important; padding:0 !important;
}
#miq-report > div { gap:0 !important; padding:0 !important; }
#miq-report .block {
    padding:0 !important; background:transparent !important;
    border:none !important; box-shadow:none !important;
}

/* Markdown prose inside report */
#miq-report-md { padding:0 !important; }
#miq-report-md > .block { padding:0 !important; border:none !important; box-shadow:none !important; }
#miq-report-md .prose { padding:20px 22px !important; }
#miq-report-md p { font-size:14.5px !important; line-height:1.78 !important;
                   color:#374151 !important; margin:0 0 10px !important; }
#miq-report-md p:last-child { margin-bottom:0 !important; }
#miq-report-md strong { color:#111827 !important; font-weight:700 !important; }
#miq-report-md h3 {
    font-size:11px !important; font-weight:800 !important; color:#9ca3af !important;
    text-transform:uppercase !important; letter-spacing:.09em !important;
    padding:16px 22px 9px !important; margin:0 !important;
    border-top:1px solid #f1f5f9 !important; border-bottom:none !important;
    background:transparent !important;
}
#miq-report-md h3:first-child { padding-top:20px !important; border-top:none !important; }
#miq-report-md ul, #miq-report-md ol { padding-left:20px !important; margin:6px 0 10px !important; }
#miq-report-md li { font-size:14.5px !important; color:#374151 !important;
                    margin:4px 0 !important; line-height:1.6 !important; }
#miq-report-md code {
    background:#f1f5f9 !important; border:1px solid #e2e8f0 !important;
    padding:1px 6px !important; border-radius:4px !important;
    font-size:13px !important; color:#374151 !important;
}
#miq-report-md a { color:#6366f1 !important; text-decoration:none !important; }
#miq-report-md table { width:100% !important; border-collapse:collapse !important;
                       margin:10px 0 !important; font-size:13.5px !important; }
#miq-report-md thead th {
    background:#f8fafc !important; border:1px solid #e2e8f0 !important;
    padding:10px 14px !important; font-size:11px !important; font-weight:800 !important;
    color:#64748b !important; text-transform:uppercase !important;
    letter-spacing:.07em !important; text-align:left !important;
}
#miq-report-md tbody td { border:1px solid #f1f5f9 !important;
                          padding:9px 14px !important; color:#374151 !important; }
#miq-report-md tbody td:first-child { font-weight:600 !important; color:#111827 !important; }
#miq-report-md tbody tr:nth-child(even) td { background:#f8fafc !important; }
#miq-report-md tbody tr:hover td { background:#eef2ff !important; }

/* ── Timing ───────────────────────────── */
#miq-timing .block {
    padding:0 !important; background:transparent !important;
    border:none !important; box-shadow:none !important;
}

/* ── Footer ───────────────────────────── */
#miq-footer { border-top:1px solid #e2e8f0 !important; background:#fff !important; }
#miq-footer > .block {
    padding:11px 24px !important; background:transparent !important;
    border:none !important; box-shadow:none !important; border-radius:0 !important;
}
"""


# ── Streaming handler ───────────────────────────────────────────────────

def stream_answer(
    question: str,
) -> Generator[tuple[str, str, str, str], None, None]:
    if not question or not question.strip():
        yield ("", _verdict_html("", "", ""), "", "")
        return

    from agents.orchestrator import run_pipeline_streaming

    start = time.time()
    progress_lines: list[str] = []
    compound_name = formula = ""

    try:
        for msg, answer in run_pipeline_streaming(question, hf_token=HF_TOKEN):
            progress_lines.append(msg)

            if not compound_name and "Molecule identified:" in msg:
                m = re.search(r"Molecule identified:\s*([^(]+)", msg)
                if m:
                    compound_name = m.group(1).strip()

            if not formula and "Molecule identified:" in msg:
                m = re.search(r"Molecule identified:[^(]+\(([^,]+),", msg)
                if m:
                    formula = m.group(1).strip()

            prog = _progress_html(progress_lines)

            if answer is not None:
                elapsed = time.time() - start

                badge_m = re.search(r"## Safety verdict:\s*(.+)", answer)
                badge = badge_m.group(1).strip() if badge_m else ""

                if not formula:
                    fm = re.search(r"\*\*Molecular formula:\*\*\s*(\S+)", answer)
                    if fm:
                        formula = fm.group(1)

                verdict = _verdict_html(badge, compound_name, formula)

                display = re.sub(
                    r"^## Safety verdict:.+\n## Toxicity risk:.+\n+",
                    "",
                    answer,
                    flags=re.MULTILINE,
                )

                timing = (
                    f'<div style="font-family:Inter,system-ui,sans-serif;'
                    f'display:flex;align-items:center;gap:6px;'
                    f'font-size:12px;color:#94a3b8;padding:2px 0;">'
                    f'⏱ {elapsed:.1f}s · pipeline complete'
                    f'</div>'
                )
                yield (prog, verdict, display, timing)
            else:
                yield (prog, _verdict_html("", "", ""), "", "")

    except Exception as exc:
        logger.error("Pipeline error for '%s': %s", question, exc)
        elapsed = time.time() - start
        yield (
            _progress_html(progress_lines + [f"Failed: {exc}"]),
            _verdict_html("", "", ""),
            f"**Error:** `{exc}`\n\nPlease try again.",
            f'<div style="font-size:12px;color:#94a3b8;">Failed after {elapsed:.1f}s</div>',
        )


# ── Interface ───────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    try:
        theme = MoleculeIQTheme()
    except Exception as exc:
        logger.warning("Custom theme failed (%s) — using built-in Soft theme", exc)
        theme = gr.themes.Soft()

    with gr.Blocks(title="MoleculeIQ", theme=theme, css=CSS) as demo:

        # ── Nav
        with gr.Row(elem_id="miq-nav"):
            gr.HTML("""
            <div style="display:flex;align-items:center;gap:10px;
                        padding:12px 24px;background:#fff;width:100%;
                        font-family:Inter,system-ui,sans-serif;">
              <div style="width:34px;height:34px;background:#1e293b;border-radius:10px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:17px;flex-shrink:0;">🧪</div>
              <span style="font-size:17px;font-weight:800;color:#0f172a;
                           letter-spacing:-.04em;">MoleculeIQ</span>
              <span style="font-size:10px;font-weight:700;letter-spacing:.06em;
                           color:#6366f1;background:#eef2ff;border:1px solid #c7d2fe;
                           padding:2px 8px;border-radius:100px;margin-left:2px;">BETA</span>
              <div style="flex:1;"></div>
              <span style="font-size:12px;font-weight:600;color:#92400e;
                           background:#fefce8;border:1px solid #fde68a;
                           padding:6px 14px;border-radius:100px;">
                ⚠ Not medical advice — consult a professional
              </span>
            </div>
            """)

        # ── Search
        with gr.Row(elem_id="miq-search-row"):
            question_input = gr.Textbox(
                show_label=False,
                placeholder="Ask about any food additive, chemical or ingredient — e.g. Is BPA harmful?",
                lines=1,
                scale=8,
                elem_id="miq-q",
                container=False,
            )
            submit_btn = gr.Button(
                "Analyse →",
                scale=1,
                min_width=140,
                elem_id="miq-btn",
                variant="primary",
            )

        # ── Chips
        with gr.Row(elem_id="miq-chips"):
            chip_btns = [
                gr.Button(q, size="sm", variant="secondary")
                for q in EXAMPLE_QUESTIONS
            ]

        # ── Body
        with gr.Row(elem_id="miq-body", equal_height=False):

            # Left: pipeline
            with gr.Column(scale=3, elem_id="miq-pipeline"):
                gr.HTML(
                    '<div style="font-family:Inter,system-ui,sans-serif;margin-bottom:16px;">'
                    '<div style="font-size:11px;font-weight:800;color:#94a3b8;'
                    'text-transform:uppercase;letter-spacing:.1em;">How we analysed this</div>'
                    '<div style="font-size:11.5px;color:#94a3b8;margin-top:4px;line-height:1.5;">'
                    'Checks 4 independent sources</div>'
                    '</div>'
                )
                progress_output = gr.HTML(value=_progress_html([]))
                timing_output   = gr.HTML(value="", elem_id="miq-timing")

            # Right: verdict + report card
            with gr.Column(scale=7, elem_id="miq-results"):
                verdict_output = gr.HTML(
                    value=_verdict_html("", "", ""),
                    elem_id="miq-verdict",
                )
                with gr.Column(elem_id="miq-report"):
                    gr.HTML(
                        '<div style="padding:12px 20px;background:#f8fafc;'
                        'border-bottom:1px solid #e2e8f0;display:flex;'
                        'align-items:center;gap:8px;">'
                        '<span style="font-size:11px;font-weight:800;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:.09em;'
                        'font-family:Inter,system-ui,sans-serif;">📄 Analysis report</span>'
                        '</div>'
                    )
                    answer_output = gr.Markdown(
                        value="",
                        elem_id="miq-report-md",
                    )

        # ── Footer
        with gr.Row(elem_id="miq-footer"):
            gr.HTML("""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        width:100%;flex-wrap:wrap;gap:8px;
                        font-family:Inter,system-ui,sans-serif;">
              <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                <span style="font-size:12px;color:#64748b;background:#f1f5f9;
                             border:1px solid #e2e8f0;padding:3px 10px;
                             border-radius:100px;">PubChem</span>
                <span style="font-size:12px;color:#64748b;background:#f1f5f9;
                             border:1px solid #e2e8f0;padding:3px 10px;
                             border-radius:100px;">OpenFDA</span>
                <span style="font-size:12px;color:#64748b;background:#f1f5f9;
                             border:1px solid #e2e8f0;padding:3px 10px;
                             border-radius:100px;">Tox21</span>
                <span style="font-size:12px;color:#64748b;background:#f1f5f9;
                             border:1px solid #e2e8f0;padding:3px 10px;
                             border-radius:100px;">Qwen2.5-7B-Instruct</span>
              </div>
              <span style="font-size:12px;color:#94a3b8;">
                Powered by Hugging Face Inference API
              </span>
            </div>
            """)

        # ── Wiring
        outputs = [progress_output, verdict_output, answer_output, timing_output]

        for btn, q in zip(chip_btns, EXAMPLE_QUESTIONS):
            btn.click(fn=lambda x=q: x, outputs=question_input)

        submit_btn.click(
            fn=stream_answer,
            inputs=question_input,
            outputs=outputs,
            api_name="ask",
        )
        question_input.submit(
            fn=stream_answer,
            inputs=question_input,
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    _prewarm_models()
    demo = build_interface()
    demo.queue(max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )