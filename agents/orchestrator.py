"""Orchestrator — smolagents CodeAgent that dispatches all sub-agents."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Generator

from smolagents import CodeAgent, InferenceClientModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.researcher import ResearcherAgent  # noqa: E402
from agents.safety_expert import SafetyExpertAgent  # noqa: E402
from agents.toxicity_checker import ToxicityCheckerAgent  # noqa: E402
from agents.odor_predictor import OdorPredictorAgent  # noqa: E402
from agents.reviewer import ReviewerAgent  # noqa: E402
from agents.formatter import FormatterAgent  # noqa: E402

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXTRACTOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """\
You are an ingredient safety research orchestrator. The user will ask a
question about an ingredient, food additive, or chemical compound. You must:

1. Call `researcher` to fetch PubChem background data for the molecule.
2. Call `safety_expert` with the molecule name, question, and context from
   step 1 (description, formula, cas).
3. Call `toxicity_checker` with the SMILES string returned by the researcher.
4. Call `odor_predictor` with the SMILES string returned by the researcher.
5. Call `reviewer` with the safety answer, toxicity result, and description.
6. Call `formatter` with all collected outputs.

Return ONLY the final formatted markdown from the formatter.
If any step fails, note the failure and continue with remaining steps.
Do not invent data — only use what the tools return.
"""

_EXTRACTOR_SYSTEM = (
    "Extract the chemical compound, food ingredient, or molecule name from "
    "the user's question. Return the canonical or scientific name — "
    "prefer IUPAC or common scientific names over trade names or E-numbers. "
    "Always resolve E-numbers to their chemical name. "
    "Return ONLY the compound name — no explanation, no punctuation. "
    "Examples: 'caffeine', 'allura red' (not 'Red 40' or 'E129'), "
    "'bisphenol a' (not 'BPA'), "
    "'acetaminophen' (not 'Tylenol' or 'Paracetamol'), "
    "'monosodium glutamate' (not 'MSG' or 'E621'), "
    "'sucralose' (not 'Splenda' or 'E955'), "
    "'citric acid' (not 'E330'), "
    "'titanium dioxide' (not 'E171'), "
    "'sodium nitrite' (not 'E250')."
)


def _extract_molecule_name(
    question: str, hf_token: str = ""
) -> str:
    """Extract the canonical molecule/ingredient name from the user's question."""
    token = hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for molecule name extraction. "
            "Set the HF_TOKEN environment variable."
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=EXTRACTOR_MODEL, token=token)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": _EXTRACTOR_SYSTEM},
            {"role": "user",   "content": question},
        ],
        max_tokens=20,
        temperature=0.0,
    )
    name = (
        resp.choices[0].message.content
        .strip().strip('"').strip("'").rstrip(".")
    )
    if not (1 < len(name) < 80):
        raise ValueError(
            f"Molecule name extraction returned an unusable result: {name!r}"
        )
    return name


def build_agent(hf_token: str | None = None) -> CodeAgent:
    """Build and return a smolagents CodeAgent with all sub-agent tools."""
    token = hf_token or os.getenv("HF_TOKEN", "")
    model = InferenceClientModel(
        model_id=ORCHESTRATOR_MODEL,
        token=token or None,
    )
    tools = [
        ResearcherAgent(),
        SafetyExpertAgent(),
        ToxicityCheckerAgent(),
        OdorPredictorAgent(),
        ReviewerAgent(),
        FormatterAgent(),
    ]
    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=12,
        additional_authorized_imports=["json", "re"],
        system_prompt=SYSTEM_PROMPT,
    )


def run_pipeline_streaming(
    question: str,
    hf_token: str | None = None,
) -> Generator[tuple[str, str | None], None, None]:
    """
    Run the full multi-agent pipeline with step-by-step progress.

    Yields (progress_message, answer_or_None) tuples. The final tuple
    has a non-None answer containing the complete markdown response.
    """
    from agents.researcher import run_researcher
    from agents.safety_expert import run_safety_expert
    from agents.toxicity_checker import run_toxicity_checker
    from agents.odor_predictor import run_odor_predictor
    from agents.reviewer import run_reviewer
    from agents.formatter import run_formatter

    molecule_name = _extract_molecule_name(question, hf_token or "")
    yield (f"Researching {molecule_name}...", None)

    # Step 1 — Researcher
    researcher_data: dict[str, Any] = {}
    try:
        researcher_data = run_researcher(molecule_name)
        cid = researcher_data.get("cid", "N/A")
        formula = (
            researcher_data.get("molecular_formula", "") or molecule_name
        )
        # Normalize to PubChem IUPAC name so phrasing variants
        # ("alcohol" vs "ethanol") hit identical safety context.
        iupac = researcher_data.get("iupac_name", "")
        if iupac and len(iupac) < 50:
            molecule_name = iupac
        display_name = researcher_data.get("name", "") or molecule_name
        yield (
            f"Molecule identified: {display_name} ({formula}, PubChem CID: {cid})", None
        )
    except Exception as exc:
        logger.error("Researcher failed: %s", exc)
        researcher_data = {
            "name": molecule_name, "smiles": "", "source": "error"
        }
        yield ("Researcher failed — continuing without PubChem data", None)

    # Step 2 — Safety Expert (grounded: pass researcher context)
    yield ("Running safety analysis...", None)
    safety_answer = ""
    try:
        safety_answer = run_safety_expert(
            molecule_name,
            question,
            description=researcher_data.get("description", ""),
            formula=researcher_data.get("molecular_formula", ""),
            cas=researcher_data.get("cas_number", ""),
            hf_token=hf_token,
        )
        yield ("Safety analysis complete", None)
    except Exception as exc:
        logger.error("SafetyExpert failed: %s", exc)
        safety_answer = f"Safety analysis unavailable ({exc})."
        yield ("Safety analysis failed — continuing", None)

    # Step 3 — Toxicity Checker
    yield ("Running toxicity prediction...", None)
    smiles = researcher_data.get("smiles", "")
    toxicity_result: dict[str, Any] = {
        "risk_level": "UNKNOWN", "assay_scores": {}
    }
    if smiles:
        try:
            toxicity_result = run_toxicity_checker(smiles)
            risk = toxicity_result.get("risk_level", "UNKNOWN")
            yield (f"Toxicity check complete — {risk} risk", None)
        except Exception as exc:
            logger.error("ToxicityChecker failed: %s", exc)
            toxicity_result = {
                "risk_level": "UNKNOWN",
                "assay_scores": {},
                "error": str(exc),
            }
            yield ("Toxicity prediction failed — continuing", None)
    else:
        toxicity_result["error"] = "No SMILES available"
        yield ("No SMILES — skipping toxicity check", None)

    # Step 4 — Odor Predictor
    odor_data: dict[str, Any] = {"odors": {}, "top_odor": "unknown"}
    if smiles:
        yield ("Predicting odor profile...", None)
        try:
            odor_data = run_odor_predictor(smiles)
            top = odor_data.get("top_odor", "unknown")
            n = len(odor_data.get("odors", {}))
            yield (f"Odor profile: {top} ({n} descriptors detected)", None)
        except Exception as exc:
            logger.error("OdorPredictor failed: %s", exc)
            odor_data = {
                "odors": {}, "top_odor": "unknown", "error": str(exc)
            }
            yield ("Odor prediction failed — continuing", None)
    else:
        yield ("No SMILES — skipping odor prediction", None)

    # Step 5 — Reviewer (LLM-based, with molecule context)
    yield ("Cross-checking consistency...", None)
    reviewer_result: dict[str, Any] = {}
    try:
        reviewer_result = run_reviewer(
            safety_answer,
            toxicity_result,
            description=researcher_data.get("description", ""),
            hf_token=hf_token,
        )
        verdict = reviewer_result.get("verdict", "inconclusive")
        yield (f"Review verdict: {verdict}", None)
    except Exception as exc:
        logger.error("Reviewer failed: %s", exc)
        reviewer_result = {
            "verdict": "inconclusive",
            "explanation": f"Review failed: {exc}",
        }
        yield ("Review step failed — marking inconclusive", None)

    # Step 6 — Formatter
    yield ("Formatting final answer...", None)
    try:
        final_markdown = run_formatter(
            researcher_data=researcher_data,
            safety_answer=safety_answer,
            toxicity_result=toxicity_result,
            odor_data=odor_data,
            reviewer_result=reviewer_result,
        )
        yield ("Done!", final_markdown)
    except Exception as exc:
        logger.error("Formatter failed: %s", exc)
        fallback = (
            f"**{molecule_name.title()} — Safety Summary**\n\n"
            f"{safety_answer}\n\n"
            f"**Toxicity risk:** "
            f"{toxicity_result.get('risk_level', 'Unknown')}\n\n"
            f"**Review verdict:** "
            f"{reviewer_result.get('verdict', 'inconclusive')}"
        )
        yield ("Formatter failed — showing plain summary", fallback)
