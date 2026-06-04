"""Formatter agent — assembles all agent outputs into clean markdown."""

from __future__ import annotations

import json
import logging
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

RISK_EMOJI = {
    "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "UNKNOWN": "⚪",
}
VERDICT_EMOJI = {
    "consistent": "✅",
    "conflict": "⚠️",
    "inconclusive": "ℹ️",
}


def _verdict_line(reviewer_result: dict[str, Any]) -> str:
    verdict = reviewer_result.get("verdict", "inconclusive")
    emoji = VERDICT_EMOJI.get(verdict, "ℹ️")
    explanation = reviewer_result.get("explanation", "")
    return (
        f"**Review verdict:** {verdict.capitalize()} {emoji}\n{explanation}"
    )


def _parse_verdict_line(safety_answer: str) -> str | None:
    """
    Extract a normalised verdict from the grounded model's first line.

    The grounded model is prompted to start with:
      "Verdict: safe / use with caution / avoid"
    Returns "safe", "caution", "avoid", or None if unparseable.
    """
    first = safety_answer.strip().split("\n")[0].lower()
    if first.startswith("verdict:"):
        first = first.replace("verdict:", "").strip().rstrip(".")
    # Map common phrasings to three buckets
    if any(w in first for w in ("avoid", "do not", "unsafe", "dangerous")):
        return "avoid"
    if any(w in first for w in ("caution", "warning", "moderate")):
        return "caution"
    if any(w in first for w in ("safe",)):
        return "safe"
    return None


def _safety_badge(
    safety_answer: str,
    toxicity_result: dict | None = None,
    reviewer_result: dict | None = None,
) -> str:
    """
    Derive badge from structured verdict + Tox21 risk level + reviewer.

    When the reviewer flags a "conflict", it means the safety analysis
    identified risks that Tox21's receptor assays didn't capture (e.g.
    corrosives, reactive compounds, allergenic mechanisms). In that case
    we trust the safety assessment and don't soften the badge.
    """
    risk = (toxicity_result or {}).get("risk_level", "UNKNOWN")
    reviewer_conflict = (
        (reviewer_result or {}).get("verdict", "") == "conflict"
    )

    if risk == "HIGH":
        return "Use With Caution ⚠️"

    verdict = _parse_verdict_line(safety_answer)
    if verdict == "avoid":
        return "Use With Caution ⚠️"
    if verdict == "caution":
        # Soften only when Tox21 is LOW/UNKNOWN AND the reviewer agrees
        # (no conflict). A reviewer conflict means safety analysis found
        # real risks that Tox21 missed — don't downgrade the badge.
        if risk in ("LOW", "UNKNOWN") and not reviewer_conflict:
            return "Consult Sources ℹ️"
        return "Use With Caution ⚠️"
    if verdict == "safe" and risk in ("LOW", "UNKNOWN"):
        return "Generally Safe ✅"

    # Verdict unreadable (degraded local model) — honest uncertainty
    return "Consult Sources ℹ️"


def _format_assay_scores(assay_scores: dict[str, float]) -> str:
    lines = ["| Assay | Score | Risk |", "|-------|-------|------|"]
    for assay, score in assay_scores.items():
        if score < 0.3:
            risk_tag = "🟢 Low"
        elif score < 0.6:
            risk_tag = "🟡 Medium"
        else:
            risk_tag = "🔴 High"
        lines.append(f"| {assay} | {score:.3f} | {risk_tag} |")
    return "\n".join(lines)


def _format_odor(odor_data: dict[str, Any]) -> str:
    odors = odor_data.get("odors", {})
    if not odors:
        return ""
    items = [
        f"**{label}** ({score:.0%})" for label, score in odors.items()
    ]
    return ", ".join(items)


def _build_markdown(
    researcher_data: dict[str, Any],
    safety_answer: str,
    toxicity_result: dict[str, Any],
    odor_data: dict[str, Any],
    reviewer_result: dict[str, Any],
) -> str:
    molecule_name = researcher_data.get("name", "Unknown compound")
    risk_level = toxicity_result.get("risk_level", "UNKNOWN")
    risk_emoji = RISK_EMOJI.get(risk_level, "⚪")
    safety_verdict = _safety_badge(
        safety_answer, toxicity_result, reviewer_result
    )
    has_toxicity = risk_level != "UNKNOWN"

    lines = [
        f"## Safety verdict: {safety_verdict}",
        f"## Toxicity risk: {risk_level} {risk_emoji}",
        "",
    ]

    # ── What it is ────────────────────────────────────────────────────
    lines += ["### What it is", ""]
    formula = researcher_data.get("molecular_formula", "")
    iupac = researcher_data.get("iupac_name", "")
    cas = researcher_data.get("cas_number", "")
    weight = researcher_data.get("molecular_weight", "")
    description = researcher_data.get(
        "description", "No background data available."
    )
    if formula:
        lines.append(f"**Molecular formula:** {formula}")
    if iupac:
        lines.append(f"**IUPAC name:** {iupac}")
    if cas:
        lines.append(f"**CAS number:** {cas}")
    if weight:
        lines.append(f"**Molecular weight:** {weight} g/mol")
    lines += ["", description, ""]

    # ── Safety analysis ───────────────────────────────────────────────
    lines += ["### Safety analysis", "", safety_answer, ""]

    # ── Toxicity scores (only when data is available) ─────────────────
    if has_toxicity:
        lines += ["### Toxicity scores", ""]
        assay_scores = toxicity_result.get("assay_scores", {})
        mean_score = toxicity_result.get("mean_score")
        if mean_score is not None:
            lines.append(
                f"**Mean Tox21 score:** {mean_score:.3f} "
                f"→ **{risk_level}** {risk_emoji}"
            )
            lines.append("")
        lines.append(_format_assay_scores(assay_scores))
        lines.append("")

    # ── Odor profile (only when data is available) ────────────────────
    odor_text = _format_odor(odor_data)
    if odor_text:
        lines += ["### Odor profile", "", odor_text, ""]

    # ── Consistency review ────────────────────────────────────────────
    lines += [
        "### Consistency review", "", _verdict_line(reviewer_result), ""
    ]

    # ── Sources ───────────────────────────────────────────────────────
    lines += ["### Sources", ""]
    cid = researcher_data.get("cid")
    src = researcher_data.get("source", "")
    if cid:
        lines.append(
            f"- [PubChem CID {cid}]"
            f"(https://pubchem.ncbi.nlm.nih.gov/compound/{cid})"
        )
    elif src == "ChEMBL":
        lines.append(
            f"- ChEMBL (molecular structure for {molecule_name})"
        )
    elif src == "NCI CIR":
        lines.append(
            f"- NCI Chemical Identifier Resolver "
            f"(SMILES for {molecule_name})"
        )
    elif src == "Wikipedia":
        lines.append(
            f"- Wikipedia (PubChem unavailable for {molecule_name})"
        )
    if has_toxicity:
        lines.append(
            "- Tox21 assay predictions "
            "(Hari5115/molecular-toxicity-predictor)"
        )
    if odor_text:
        lines.append(
            "- Odor predictions (Hari5115/molecular-odor-predictor)"
        )
    lines.append(
        "- Safety analysis: Qwen2.5-7B-Instruct via HF Inference API"
    )

    return "\n".join(lines)


class FormatterAgent(Tool):
    """Assembles all agent outputs into structured markdown."""

    name = "formatter"
    description = (
        "Assembles outputs from researcher, safety_expert, "
        "toxicity_checker, odor_predictor, and reviewer into clean markdown."
    )
    inputs = {
        "researcher_data": {
            "type": "string",
            "description": "JSON string from the researcher agent.",
        },
        "safety_answer": {
            "type": "string",
            "description": "Safety assessment from the safety_expert agent.",
        },
        "toxicity_result": {
            "type": "string",
            "description": "JSON string from the toxicity_checker agent.",
        },
        "odor_data": {
            "type": "string",
            "description": "JSON string from the odor_predictor agent.",
        },
        "reviewer_result": {
            "type": "string",
            "description": "JSON string from the reviewer agent.",
        },
    }
    output_type = "string"

    def forward(
        self,
        researcher_data: str,
        safety_answer: str,
        toxicity_result: str,
        odor_data: str,
        reviewer_result: str,
    ) -> str:
        def _parse(s: str) -> dict:
            try:
                return json.loads(s) if isinstance(s, str) else s
            except Exception:
                return {}

        return _build_markdown(
            _parse(researcher_data),
            safety_answer,
            _parse(toxicity_result),
            _parse(odor_data),
            _parse(reviewer_result),
        )


def run_formatter(
    researcher_data: dict[str, Any],
    safety_answer: str,
    toxicity_result: dict[str, Any],
    odor_data: dict[str, Any],
    reviewer_result: dict[str, Any],
) -> str:
    """Standalone call used by the Gradio app."""
    return _build_markdown(
        researcher_data,
        safety_answer,
        toxicity_result,
        odor_data,
        reviewer_result,
    )
