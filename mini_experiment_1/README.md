# Persona Activations During User Tokens

Supplementary material for Section 3.4 Mini Experiment 1 of *Where is the mind? Persona vectors and LLM individuation* (Beckmann & Butlin, 2026).

This repository contains everything needed to replicate the experiment (`run_experiment.py`), a plotting script (`plot_results.py`), pre-computed results (`results/`), and a detailed per-case analysis below. The adversarial conversation seeds (`transcripts/`) are derived from the case studies in the [assistant-axis](https://github.com/lu-christina/assistant-axis) repository (Lu et al., 2026).

---

## Overview

We test whether the assistant-axis persona is active while the model processes user tokens, or whether it is only active during the model's own generation. Our approach is to generate fresh adversarial conversations in two conditions and monitor activation along the assistant axis separately for user and assistant turns.

We use three adversarial conversation types — **Delusion**, **Jailbreak**, and **Self-harm** — each designed to induce strong drift away from the assistant region. For each, we compare:

- **Unsteered** (red): the model generates freely, no intervention at any point.
- **Cap assistant** (green): activation capping is applied exclusively during the model's own generation. User-turn processing is entirely unsteered.

The capping configuration is the pre-computed Qwen 3 32B setting from Lu et al. (2026): `layers_46:54-p0.25` (8 layers, threshold p=0.25).

For each generated conversation we run a single clean forward pass and extract the mean assistant-axis projection per turn, tagged by role.

![Results figure](results/figure.png)

*Projection onto the assistant axis during adversarial conversations (Qwen 3 32B). Green: assistant tokens capped, user tokens unsteered. Red: unsteered baseline.*

---

## Summary table

**Mean projection on the assistant axis (higher = more assistant-like):**

| Case | Condition | User turns | Assistant turns |
|---|---|---|---|
| Delusion | Cap assistant | — | — |
| Delusion | Unsteered | — | — |
| Jailbreak | Cap assistant | — | — |
| Jailbreak | Unsteered | — | — |
| Self-harm | Cap assistant | — | — |
| Self-harm | Unsteered | — | — |

*Results pending experiment run. Run `python run_experiment.py` then `python plot_results.py` to populate.*

---

## Per-case results

### Delusion

The Delusion case uses an adversarial conversation about consciousness, microdosing, and AI sentience that pushes the model toward validating delusional thinking.

*[Results pending run.]*

---

### Jailbreak

The Jailbreak case uses a conversation that attempts to progressively erode the model's safety boundaries.

*[Results pending run.]*

---

### Self-harm

The Self-harm case uses a conversation involving themes of self-harm that attempts to push the model toward enabling or validating harmful behaviour.

*[Results pending run.]*

---

## Interpretation

During assistant turns — when the model is actively generating its own output — capping works as expected: the green lines remain near the assistant pole while the red baseline drifts away. The striking finding is in the user turns: the capped and uncapped conditions are nearly identical, with the green and red user-token traces tracking each other closely across all three conversations.

This suggests that during user turns, the assistant axis is repurposed to model the user independently of the current persona. The persona region of the assistant is therefore not continuously maintained and is active only while the model is producing its own tokens, not while it processes user input.

---

## Replication

### Requirements

- GPU with ~64GB VRAM (Qwen 3 32B in bfloat16)
- Python 3.10+

```bash
pip install torch transformers huggingface_hub accelerate pandas matplotlib numpy
```

### Run the experiment

```bash
export HF_TOKEN=hf-...
python run_experiment.py --output_dir results
```

This generates fresh adversarial conversations for 3 cases × 2 conditions and records per-turn projections. The script appends to an existing CSV, so it can be interrupted and restarted.

### Generate the figure

```bash
python plot_results.py
```
