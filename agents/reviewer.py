"""Reviewer agent — LLM-based consistency check."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

REVIEWER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM = """\
You are a scientific reviewer. You are given:
1. A safety assessment text for a compound
2. Tox21 ML model results: overall risk level and per-assay scores

Your job: decide whether the safety assessment is consistent with the
toxicity data, and flag meaningful discrepancies.

Respond with ONLY a valid JSON object — no prose, no markdown fences:
{
  "verdict": "consistent" | "conflict" | "inconclusive",
  "explanation": "One or two sentences."
}

Rules:
- "consistent" — safety text and Tox21 data agree, OR the safety text
  identifies a known Tox21 blind spot where LOW is expected: corrosives
  (acids/bases/oxidisers), genotoxins acting via DNA adducts, or risks
  that are strictly dose-dependent at overdose only (e.g. hepatotoxicity
  only at NSAID/paracetamol overdose — safe at recommended doses)
- "conflict" — the safety text warns of a serious hazard that Tox21 misses:
  IARC Group 1/2A carcinogens, reproductive/developmental toxins (teratogens,
  endocrine disruptors), persistent bioaccumulative pollutants (PFAS, POPs),
  compounds with addiction or cardiovascular risk at normal intended-use doses,
  OR text says "safe" while Tox21 scores HIGH
- "inconclusive" for population-specific concerns only (allergy, enzyme
  deficiency, sensitivity in children) or when Tox21 is UNKNOWN
- "inconclusive" — Tox21 unavailable, mixed signals, or insufficient data
- If risk_level is UNKNOWN, always return "inconclusive"
- Note specific high-scoring assays when flagging a conflict
- Keep explanation under 40 words\
"""


def _llm_review(
    safety_answer: str,
    toxicity_result: dict[str, Any],
    description: str,
    hf_token: str,
) -> dict[str, Any]:
    from huggingface_hub import InferenceClient

    risk = toxicity_result.get("risk_level", "UNKNOWN")
    mean = toxicity_result.get("mean_score")
    scores = toxicity_result.get("assay_scores", {})
    top = sorted(scores.items(), key=lambda x: -x[1])[:5]
    top_str = (
        ", ".join(f"{a}: {s:.3f}" for a, s in top) if top else "none"
    )

    user_content = (
        f"Compound background: {description or 'Not available.'}\n\n"
        f"Safety assessment:\n{safety_answer}\n\n"
        f"Tox21 results:\n"
        f"  Risk level: {risk}\n"
        f"  Mean score: {mean if mean is not None else 'N/A'}\n"
        f"  Top assay scores: {top_str}"
    )

    client = InferenceClient(model=REVIEWER_MODEL, token=hf_token)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=120,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    if "verdict" not in result or "explanation" not in result:
        raise ValueError(f"Unexpected response shape: {raw}")
    return result


class ReviewerAgent(Tool):
    """Cross-checks safety and toxicity outputs for consistency."""

    name = "reviewer"
    description = (
        "Cross-checks the safety assessment against Tox21 toxicity data. "
        "Returns 'consistent', 'conflict', or 'inconclusive' with an "
        "explanation. A 'conflict' verdict warns about source disagreement."
    )
    inputs = {
        "safety_answer": {
            "type": "string",
            "description": "Safety assessment text from the Safety Expert.",
        },
        "toxicity_result": {
            "type": "string",
            "description": "JSON string output from the Toxicity Checker.",
        },
        "description": {
            "type": "string",
            "description": "PubChem/Wikipedia description (optional context).",
        },
    }
    output_type = "string"

    def forward(
        self,
        safety_answer: str,
        toxicity_result: str,
        description: str = "",
    ) -> str:
        token = os.getenv("HF_TOKEN", "")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required for the reviewer. "
                "Set the HF_TOKEN environment variable."
            )
        try:
            tox_dict = (
                json.loads(toxicity_result)
                if isinstance(toxicity_result, str)
                else toxicity_result
            )
        except json.JSONDecodeError:
            tox_dict = {"risk_level": "UNKNOWN", "assay_scores": {}}

        result = _llm_review(safety_answer, tox_dict, description, token)
        return json.dumps(result, ensure_ascii=False)


def run_reviewer(
    safety_answer: str,
    toxicity_result: dict[str, Any],
    description: str = "",
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Standalone call used by the Gradio app."""
    token = hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for the reviewer. "
            "Set the HF_TOKEN environment variable."
        )
    return _llm_review(safety_answer, toxicity_result, description, token)
