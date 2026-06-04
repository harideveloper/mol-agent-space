"""Odor Predictor agent — predicts smell profile from SMILES using MLP."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

MODEL_REPO = "Hari5115/molecular-odor-predictor"
ODOR_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def _load_model():
    """Download and cache the odor MLP model and labels from HF Hub."""
    import torch
    import torch.nn as nn  # noqa: F401 — used in OdorMLP class body
    from huggingface_hub import hf_hub_download

    labels_path = hf_hub_download(repo_id=MODEL_REPO, filename="labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    class OdorMLP(nn.Module):
        def __init__(
            self, n_inputs: int, n_outputs: int, dropout: float = 0.4
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_inputs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_outputs),
            )

        def forward(self, x):
            return self.net(x)

    model = OdorMLP(2048, len(labels))
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pt")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, labels


def _run_prediction(smiles: str) -> dict[str, Any]:
    """Predict odor descriptors for a molecule given its SMILES string."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles!r}", "odors": {}}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = torch.tensor(
        [list(gen.GetFingerprint(mol))], dtype=torch.float32
    )

    model, labels = _load_model()
    with torch.no_grad():
        probs = torch.sigmoid(model(fp)).squeeze().numpy()

    all_scores = {
        label: round(float(p), 3)
        for label, p in zip(labels, probs)
    }
    detected = {
        label: score
        for label, score in sorted(
            all_scores.items(), key=lambda x: -x[1]
        )
        if score >= ODOR_THRESHOLD
    }

    return {
        "smiles": smiles,
        "odors": detected,
        "top_odor": next(iter(detected), "no dominant odor"),
        "error": None,
    }


class OdorPredictorAgent(Tool):
    """Smolagents Tool that predicts odor profile from a SMILES string."""

    name = "odor_predictor"
    description = (
        "Predicts the smell profile of a molecule given its SMILES string. "
        "Returns the top odor descriptors (e.g. fruity, floral, pungent) "
        "with confidence scores. Requires a valid SMILES string."
    )
    inputs = {
        "smiles": {
            "type": "string",
            "description": "IsomericSMILES string for the molecule.",
        }
    }
    output_type = "string"

    def forward(self, smiles: str) -> str:
        if not smiles or not smiles.strip():
            return json.dumps({
                "error": "No SMILES provided",
                "odors": {},
                "top_odor": "unknown",
            })
        try:
            result = _run_prediction(smiles.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.error(
                "OdorPredictorAgent failed for SMILES '%s': %s", smiles, exc
            )
            return json.dumps({
                "error": str(exc),
                "odors": {},
                "top_odor": "unknown",
            })


def run_odor_predictor(smiles: str) -> dict[str, Any]:
    """Standalone call used by the Gradio app."""
    agent = OdorPredictorAgent()
    return json.loads(agent.forward(smiles))
