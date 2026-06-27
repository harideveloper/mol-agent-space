---
title: MoleculeIQ Agent
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.9.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - smolagents
  - chemistry
  - safety
  - multi-agent
  - ingredient-safety
  - toxicology
  - gradio
---

# MoleculeIQ Agent 🧪

**Multi-agent molecular safety Q&A powered by smolagents.**

Ask about any ingredient, food additive, or chemical compound and get a comprehensive,
sourced safety assessment in seconds.

> ⚠️ For educational purposes only. Not medical advice — always consult a professional.

---

## What it does

MoleculeIQ orchestrates 6 specialised agents to answer ingredient safety questions:

| Agent | Role |
|-------|------|
| Researcher | Fetches molecular data from PubChem (formula, CAS, SMILES, description) |
| Safety Expert | LLM-grounded safety analysis via Qwen2.5-7B-Instruct |
| Toxicity Checker | Predicts Tox21 assay scores (12 receptor pathways) |
| Odor Predictor | Predicts odor descriptors from molecular structure |
| Reviewer | Cross-checks safety analysis against toxicity data for conflicts |
| Formatter | Assembles clean markdown with verdict badge, scores, and sources |

---

## Example questions

- Is aspartame safe to consume daily?
- Is bleach dangerous?
- Is BPA harmful?
- Is ibuprofen safe long-term?
- Is alcohol safe daily?
- Is lead dangerous?

---

## Architecture

```
User question
      ↓
  Name extractor (Qwen2.5-7B — resolves E-numbers, trade names)
      ↓
  Researcher → Safety Expert → Toxicity Checker → Odor Predictor → Reviewer → Formatter
      ↓
  Verdict badge + toxicity scores + odor profile + consistency review + sources
```

---

## Models

| Role | Model |
|------|-------|
| Name extraction + safety analysis + review | `Qwen/Qwen2.5-7B-Instruct` via HF Inference API |
| Toxicity prediction (Tox21) | [`Hari5115/molecular-toxicity-predictor`](https://huggingface.co/Hari5115/molecular-toxicity-predictor) |
| Odor prediction | [`Hari5115/molecular-odor-predictor`](https://huggingface.co/Hari5115/molecular-odor-predictor) |

---

## Data sources

- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) — molecular identity, formula, CAS, SMILES
- [Tox21](https://tox21.gov/) — 12-assay toxicity screening dataset
- HF Inference API — LLM safety reasoning

---

## Limitations

- Response time: ~15–30s (6 sequential agent calls via HF Inference API)
- Tox21 model covers receptor-based pathways — may underestimate corrosives, genotoxins
- Safety analysis is LLM-generated and may contain errors
- Coverage best for food additives, pharmaceuticals, and common industrial chemicals
- Not a substitute for professional medical or toxicological advice

---

## Setup (self-hosted)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # required — used for Qwen2.5-7B inference
python app.py
```

On HuggingFace Spaces, set `HF_TOKEN` as a **Space Secret** in the Space settings.

---

_MoleculeIQ Agent · Built with [smolagents](https://github.com/huggingface/smolagents) · [Hari5115](https://huggingface.co/Hari5115)_
