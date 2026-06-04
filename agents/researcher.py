"""Researcher agent — fetches molecule background from PubChem."""

from __future__ import annotations

import json
import sys
import os

from smolagents import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.pubchem_tool import fetch_pubchem_data


class ResearcherAgent(Tool):
    """Smolagents Tool that fetches molecular background data from PubChem."""

    name = "researcher"
    description = (
        "Fetches molecular background from PubChem for a given ingredient or chemical name. "
        "Returns the molecular formula, IUPAC name, CAS number, SMILES string, and a "
        "short description. Use this as the first step before calling other agents."
    )
    inputs = {
        "molecule_name": {
            "type": "string",
            "description": "Common name, IUPAC name, or trade name of the molecule/ingredient.",
        }
    }
    output_type = "string"

    def forward(self, molecule_name: str) -> str:
        """Fetch PubChem data and return as a JSON string."""
        data = fetch_pubchem_data(molecule_name)
        return json.dumps(data, ensure_ascii=False)


def run_researcher(molecule_name: str) -> dict:
    """
    Standalone function to run the researcher agent outside of smolagents.
    Returns the parsed dict directly (used by the Gradio app for progress streaming).
    """
    agent = ResearcherAgent()
    result_json = agent.forward(molecule_name)
    return json.loads(result_json)
