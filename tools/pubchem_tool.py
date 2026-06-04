"""PubChem + ChEMBL + NCI CIR tool for fetching molecular data and SMILES."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule"
NCI_CIR_BASE = "https://cactus.nci.nih.gov/chemical/structure/{name}/{repr}"
WIKIPEDIA_SUMMARY = (
    "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
)

_MAX_RETRIES = 3
_RETRY_DELAY = 1.5   # fixed delay between retries (handles brief throttling)
_PUBCHEM_INTER_CALL_DELAY = 0.3   # pause between sequential PubChem calls
_HEADERS = {
    "User-Agent": (
        "MoleculeIQ/1.0 "
        "(research; hariprasad.sundharesan@gmail.com)"
    )
}


def fetch_pubchem_data(molecule_name: str) -> dict[str, Any]:
    """
    Fetch compound data by name.

    Source priority:
      1. PubChem  — full data (SMILES, formula, weight, CAS, description)
      2. ChEMBL   — SMILES + properties (drugs/bioactives)
      3. NCI CIR  — SMILES only (broader coverage: food additives, colorants)
      4. Wikipedia — description fallback
    """
    molecule_name = molecule_name.strip()

    # ── Step 1: PubChem ───────────────────────────────────────────────
    pubchem = _try_pubchem(molecule_name)
    if pubchem and pubchem.get("smiles"):
        return pubchem

    partial_desc = (pubchem or {}).get("description", "")

    # ── Step 2: ChEMBL (SMILES + molecular properties) ───────────────
    chembl = _try_chembl(molecule_name)

    # ── Step 3: NCI CIR (SMILES only — covers food colorants, etc.) ──
    cir_smiles = ""
    if not chembl:
        cir_smiles = _try_cir(molecule_name)

    # ── Step 4: Wikipedia (description) ──────────────────────────────
    description = partial_desc
    if not description:
        wiki = _wikipedia_fallback(molecule_name)
        description = wiki.get("description", "")

    if chembl:
        return {
            "cid": None,
            "name": molecule_name,
            "iupac_name": "",
            "molecular_formula": chembl.get("molecular_formula", ""),
            "molecular_weight": chembl.get("molecular_weight", ""),
            "cas_number": "",
            "smiles": chembl["smiles"],
            "inchikey": chembl.get("inchikey", ""),
            "description": description,
            "source": "ChEMBL",
        }

    if cir_smiles:
        return {
            "cid": None,
            "name": molecule_name,
            "iupac_name": "",
            "molecular_formula": "",
            "molecular_weight": "",
            "cas_number": "",
            "smiles": cir_smiles,
            "inchikey": "",
            "description": description,
            "source": "NCI CIR",
        }

    return {
        "cid": None,
        "name": molecule_name,
        "iupac_name": "",
        "molecular_formula": "",
        "molecular_weight": "",
        "cas_number": "",
        "smiles": "",
        "inchikey": "",
        "description": description or "No background data available.",
        "source": "Wikipedia" if description else "none",
    }


# ── Internal helpers ──────────────────────────────────────────────────

def _get(url: str, timeout: int = 10) -> requests.Response | None:
    """GET with fixed-delay retry on PubChem 503 rate-limit."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=timeout, headers=_HEADERS)
            if resp.status_code == 503:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
                    continue
                return None
            return resp
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.warning("Request failed after retries: %s", exc)
            else:
                time.sleep(_RETRY_DELAY)
    return None


def _try_pubchem(molecule_name: str) -> dict[str, Any] | None:
    """Return full PubChem data dict, or None on any failure."""
    try:
        encoded = requests.utils.quote(molecule_name)
        cid_resp = _get(
            f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
        )
        if cid_resp is None or not cid_resp.ok:
            return None

        cids = (
            cid_resp.json()
            .get("IdentifierList", {})
            .get("CID", [])
        )
        if not cids:
            return None

        cid = cids[0]
        time.sleep(_PUBCHEM_INTER_CALL_DELAY)

        props = (
            "IUPACName,MolecularFormula,MolecularWeight,"
            "IsomericSMILES,InChIKey"
        )
        prop_resp = _get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
        )
        if prop_resp is None or not prop_resp.ok:
            return None

        prop = (
            prop_resp.json()
            .get("PropertyTable", {})
            .get("Properties", [{}])[0]
        )

        time.sleep(_PUBCHEM_INTER_CALL_DELAY)
        cas = ""
        syn_resp = _get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        )
        if syn_resp and syn_resp.ok:
            syns = (
                syn_resp.json()
                .get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", [])
            )
            pat = re.compile(r"^\d{2,7}-\d{2}-\d$")
            for s in syns:
                if pat.match(s):
                    cas = s
                    break

        time.sleep(_PUBCHEM_INTER_CALL_DELAY)
        description = ""
        desc_resp = _get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/description/JSON"
        )
        if desc_resp and desc_resp.ok:
            for entry in (
                desc_resp.json()
                .get("InformationList", {})
                .get("Information", [])
            ):
                if entry.get("Description"):
                    description = entry["Description"]
                    break

        return {
            "cid": cid,
            "name": molecule_name,
            "iupac_name": prop.get("IUPACName", ""),
            "molecular_formula": prop.get("MolecularFormula", ""),
            "molecular_weight": prop.get("MolecularWeight", ""),
            "cas_number": cas,
            "smiles": prop.get("IsomericSMILES", ""),
            "inchikey": prop.get("InChIKey", ""),
            "description": description,
            "source": "PubChem",
        }
    except Exception as exc:
        logger.error("PubChem error for '%s': %s", molecule_name, exc)
        return None


def _try_chembl(molecule_name: str) -> dict[str, Any] | None:
    """Return SMILES + properties from ChEMBL, or None if not found."""
    try:
        r = requests.get(
            CHEMBL_BASE,
            params={
                "pref_name__iexact": molecule_name,
                "format": "json",
            },
            timeout=5,
            headers=_HEADERS,
        )
        if not r.ok:
            return None

        mols = r.json().get("molecules", [])
        if not mols:
            return None

        m = mols[0]
        structs = m.get("molecule_structures") or {}
        smiles = structs.get("canonical_smiles", "")
        if not smiles:
            return None

        props = m.get("molecule_properties") or {}
        return {
            "smiles": smiles,
            "molecular_formula": props.get("full_molformula", ""),
            "molecular_weight": props.get("full_mwt", ""),
            "inchikey": structs.get("standard_inchi_key", ""),
        }
    except Exception as exc:
        logger.warning(
            "ChEMBL lookup failed for '%s': %s", molecule_name, exc
        )
        return None


def _try_cir(molecule_name: str) -> str:
    """
    Return a SMILES string from the NCI Chemical Identifier Resolver.

    CIR resolves names, trade names, and E-numbers that ChEMBL misses
    (food colorants, additives, etc.). Returns empty string on failure.
    """
    try:
        encoded = requests.utils.quote(molecule_name)
        url = NCI_CIR_BASE.format(name=encoded, repr="smiles")
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        if resp.ok and resp.text.strip():
            smiles = resp.text.strip().splitlines()[0].strip()
            if smiles:
                return smiles
    except Exception as exc:
        logger.warning(
            "NCI CIR lookup failed for '%s': %s", molecule_name, exc
        )
    return ""


def _wikipedia_fallback(molecule_name: str) -> dict[str, Any]:
    """
    Fetch a short description from Wikipedia.

    Tries the full name first, then drops the last word on 404 —
    handles names with code suffixes (e.g. 'allura red ac' → 'allura red').
    """
    candidates = [molecule_name]
    parts = molecule_name.rsplit(" ", 1)
    if len(parts) == 2:
        candidates.append(parts[0])

    for candidate in candidates:
        try:
            title = requests.utils.quote(candidate)
            url = WIKIPEDIA_SUMMARY.format(title=title)
            resp = requests.get(url, timeout=8, headers=_HEADERS)
            if resp.ok:
                extract = resp.json().get("extract", "")[:500]
                if extract:
                    return {"description": extract, "source": "Wikipedia"}
        except Exception as exc:
            logger.warning(
                "Wikipedia lookup failed for '%s': %s", candidate, exc
            )
    return {"description": "", "source": "none"}
