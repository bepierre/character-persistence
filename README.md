# Persona Across Turns: Two Mini-Experiments

Supplementary material for Section 3.4 of *Where is the mind? Persona vectors and LLM individuation* (Beckmann & Butlin, 2026).

Two preliminary experiments probing when and how persona information is active across conversation turns. Both use **Qwen 3 32B** and the **assistant axis** from Lu et al. (2026).

---

## Mini Experiment 1 — Persona activations during user tokens

> **[`mini_experiment_1/`](mini_experiment_1/)**

We monitor activation along the assistant axis across three adversarial conversations (delusion, jailbreak, self-harm). We compare two conditions: a normal baseline and a condition in which capping is applied exclusively during the model's own generation (assistant tokens), while processing of user tokens proceeds entirely unsteered.

**Key finding**: During assistant turns, capping works as expected — the green lines remain near the assistant pole while the red baseline drifts away. The striking result is in the user turns: the capped and uncapped conditions are nearly identical, showing that the model's representation of user tokens along the assistant axis is largely independent of which persona is active when generating responses.

---

## Mini Experiment 2 — Persona persistence via attention streams

> **[`mini_experiment_2/`](mini_experiment_2/)**

We prefill the model on an Aura-laden conversation, then perform post-hoc editing of the KV cache: we steer the assistant axis direction at layers 20–25 for KV entries only, without re-running the forward passes.

Crucially, the KV edit is applied **only at assistant-token positions** in the prefix, consistent with Experiment 1's finding that user tokens do not carry active persona information.

**Key finding**: Editing a single direction in the cached activations — a post-hoc edit, not propagated through subsequent computation — is sufficient to override the contextual pull of 12 Aura-laden messages, confirming that LLMs reconstruct the current persona at least in part via attention to past persona activations stored in the KV cache.

---

## Citation

```bibtex
@misc{beckmann2026mind,
    title={Where is the mind? Persona vectors and LLM individuation},
    author={Beckmann, Pierre and Butlin, Patrick},
    year={2026},
}
```