"""Toxicity Checker agent — MLP on Tox21 assays."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

MODEL_REPO = "Hari5115/molecular-toxicity-predictor"

TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


@lru_cache(maxsize=1)
def _load_model():
    """Download and cache the toxicity MLP model from HuggingFace Hub."""
    import torch
    from huggingface_hub import hf_hub_download

    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO, filename="best_model.pt"
        )
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and not hasattr(state, "forward"):
            model = _build_mlp()
            model.load_state_dict(state)
        else:
            model = state
        model.eval()
        return ("torch", model)
    except Exception:
        pass

    logger.error("Failed to load toxicity model from %s", MODEL_REPO)
    return (None, None)


def _build_mlp():
    """Build the ToxMLP architecture: 2048 -> 1024 -> 512 -> 12 + BatchNorm."""
    import torch.nn as nn

    class ToxMLP(nn.Module):
        def __init__(self, n_inputs=2048, n_outputs=12, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_inputs, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, n_outputs),
            )

        def forward(self, x):
            return self.net(x)

    return ToxMLP()


def _get_morgan_fingerprint(smiles: str):
    """Convert SMILES to a 2048-bit Morgan fingerprint as a numpy array."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(float)


def _scores_to_risk(scores: list[float]) -> str:
    mean = sum(scores) / len(scores) if scores else 0.0
    if mean < 0.3:
        return "LOW"
    elif mean < 0.6:
        return "MEDIUM"
    return "HIGH"


def _run_prediction(smiles: str) -> dict[str, Any]:
    """Run the toxicity model on a SMILES string."""
    import torch

    fp = _get_morgan_fingerprint(smiles)
    kind, model = _load_model()

    if model is None:
        return {
            "error": "Model not available",
            "scores": {},
            "risk_level": "UNKNOWN",
        }

    if kind == "torch":
        tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()
        scores_list = out.tolist()
    else:
        return {
            "error": "Unknown model type",
            "scores": {},
            "risk_level": "UNKNOWN",
        }

    n = len(TOX21_ASSAYS)
    if len(scores_list) < n:
        scores_list.extend([0.0] * (n - len(scores_list)))
    scores_list = scores_list[:n]

    assay_scores = {
        assay: round(float(s), 4)
        for assay, s in zip(TOX21_ASSAYS, scores_list)
    }
    risk_level = _scores_to_risk(scores_list)

    return {
        "smiles": smiles,
        "assay_scores": assay_scores,
        "mean_score": round(sum(scores_list) / len(scores_list), 4),
        "risk_level": risk_level,
        "error": None,
    }


class ToxicityCheckerAgent(Tool):
    """Smolagents Tool that predicts Tox21 assay scores from SMILES."""

    name = "toxicity_checker"
    description = (
        "Predicts toxicity across 12 Tox21 assays for a molecule given its "
        "SMILES string. Returns per-assay probability scores and an overall "
        "risk level: LOW, MEDIUM, or HIGH. Requires a valid SMILES string — "
        "use the researcher agent to obtain one first."
    )
    inputs = {
        "smiles": {
            "type": "string",
            "description": "IsomericSMILES string for the molecule to check.",
        }
    }
    output_type = "string"

    def forward(self, smiles: str) -> str:
        if not smiles or not smiles.strip():
            return json.dumps({
                "error": "No SMILES provided",
                "risk_level": "UNKNOWN",
                "assay_scores": {},
            })
        try:
            result = _run_prediction(smiles.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.error(
                "ToxicityCheckerAgent failed for SMILES '%s': %s", smiles, exc
            )
            return json.dumps({
                "error": str(exc),
                "smiles": smiles,
                "risk_level": "UNKNOWN",
                "assay_scores": {},
            })


def run_toxicity_checker(smiles: str) -> dict[str, Any]:
    """Standalone call used by the Gradio app."""
    agent = ToxicityCheckerAgent()
    return json.loads(agent.forward(smiles))
