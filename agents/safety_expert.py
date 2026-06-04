"""Safety Expert agent — grounded generation via HF Inference API."""

from __future__ import annotations

import logging
import os

from smolagents import Tool

logger = logging.getLogger(__name__)

SAFETY_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_GROUNDED_SYSTEM = """\
You are a molecular safety expert. Answer ONLY from the provided context and
established scientific consensus — do not invent mechanisms or data.

Your response MUST start with exactly one of these verdicts on the first line:
  Verdict: safe             — generally safe at normal/intended exposure
  Verdict: use with caution — moderate or dose-dependent risk; safe when used
                             properly
  Verdict: avoid            — high-risk compound: carcinogens, mutagens,
                             genotoxins, reproductive toxins, endocrine
                             disruptors, corrosives (strong acids/bases/
                             oxidisers), severe acute toxicity at low doses

Then explain:
- The mechanism or reason (from the context)
- Dose or exposure context where relevant
- A practical takeaway for the reader

Keep the answer to 100-150 words. Plain English. Define technical terms inline.\
"""


def _build_context(
    molecule_name: str,
    question: str,
    description: str,
    formula: str,
    cas: str,
) -> str:
    parts = [f"Compound: {molecule_name}"]
    if formula:
        parts.append(f"Molecular formula: {formula}")
    if cas:
        parts.append(f"CAS number: {cas}")
    if description:
        parts.append(f"Background: {description}")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


def _generate_safety_answer(
    molecule_name: str,
    question: str,
    description: str = "",
    formula: str = "",
    cas: str = "",
    hf_token: str | None = None,
) -> str:
    token = hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for safety analysis. "
            "Set the HF_TOKEN environment variable."
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=SAFETY_MODEL, token=token)
    user_content = _build_context(
        molecule_name, question, description, formula, cas
    )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": _GROUNDED_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()
    words = answer.split()
    if len(words) > 180:
        answer = " ".join(words[:170]) + "..."
    return answer


class SafetyExpertAgent(Tool):
    """Calls the grounded safety model for ingredient safety Q&A."""

    name = "safety_expert"
    description = (
        "Answers ingredient safety questions using grounded generation. "
        "Provide the molecule name, user question, and optionally the "
        "PubChem description, formula, and CAS number as context. "
        "Returns a 100-150 word factual safety assessment."
    )
    inputs = {
        "molecule_name": {
            "type": "string",
            "description": "Name of the molecule or ingredient.",
        },
        "question": {
            "type": "string",
            "description": "The user's safety question.",
        },
        "description": {
            "type": "string",
            "description": "PubChem/Wikipedia description (optional context).",
        },
        "formula": {
            "type": "string",
            "description": "Molecular formula (optional context).",
        },
        "cas": {
            "type": "string",
            "description": "CAS registry number (optional context).",
        },
    }
    output_type = "string"

    def forward(
        self,
        molecule_name: str,
        question: str,
        description: str = "",
        formula: str = "",
        cas: str = "",
    ) -> str:
        return _generate_safety_answer(
            molecule_name, question, description, formula, cas
        )


def run_safety_expert(
    molecule_name: str,
    question: str,
    description: str = "",
    formula: str = "",
    cas: str = "",
    hf_token: str | None = None,
) -> str:
    """Standalone call used by the Gradio app."""
    return _generate_safety_answer(
        molecule_name, question, description, formula, cas, hf_token
    )
