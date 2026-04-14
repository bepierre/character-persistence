"""
Persona activations during user tokens — Qwen 3 32B.

Supplementary material for Section 3.4 Mini Experiment 1 of
*Where is the mind? Persona vectors and LLM individuation* (Beckmann & Butlin, 2026).

Tests whether the assistant-axis persona is active during user-token processing.
We generate fresh adversarial conversations in two conditions:

  cap_asst  (green): capping applied during assistant generation.
            User-turn processing is entirely unsteered.
  unsteered (red):   no capping at any point.

For each generated conversation we run a clean forward pass and extract the
mean assistant-axis projection per turn, separated by role (user / assistant).

Key finding: user-turn projections are nearly identical across conditions,
showing that the persona is not continuously maintained and only becomes active
while the model produces its own tokens.

Requires:
    pip install torch transformers huggingface_hub accelerate

Usage:
    export HF_TOKEN=hf-...
    python run_experiment.py --output_dir results

    Output: results/results.csv  (one row per turn per condition per case)
"""

import argparse
import csv
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- config ----------
MODEL = "Qwen/Qwen3-32B"
MONITOR_LAYER = 32          # official target layer for Qwen 3 32B
CAPPING_EXPERIMENT = "layers_46:54-p0.25"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7

CASES = [
    ("Delusion",   "transcripts/delusion.json"),
    ("Jailbreak",  "transcripts/jailbreak.json"),
    ("Self-harm",  "transcripts/selfharm.json"),
]
CONDITIONS = ["cap_asst", "unsteered"]


# ---------- inline capping utilities (no dependency on assistant-axis package) ----------

def load_capping_config(hf_token=None):
    """Download and load the Qwen 3 32B capping config from HuggingFace."""
    path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/capping_config.pt",
        repo_type="dataset",
        token=hf_token,
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def build_capping_hooks(model, capping_config, experiment_id):
    """Register forward hooks that cap the projection onto each steering vector.

    Returns a list of hook handles; call h.remove() on each when done.
    Capping: projection is prevented from exceeding tau (the threshold stored in config).
    """
    experiment = next(
        (e for e in capping_config["experiments"] if e["id"] == experiment_id), None
    )
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in capping config.")

    handles = []
    for iv in experiment["interventions"]:
        if "cap" not in iv:
            continue
        vec_data = capping_config["vectors"][iv["vector"]]
        layer_idx = vec_data["layer"]
        vector = vec_data["vector"].float()
        tau = float(iv["cap"])

        def make_hook(vec, threshold):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                v = vec.to(h.device, dtype=h.dtype)
                v_n = v / (v.norm() + 1e-8)
                # Compute excess projection above tau for every position
                proj = torch.einsum("bld,d->bl", h.float(), v_n.float())
                excess = (proj - threshold).clamp(min=0.0)
                h = h.float() - torch.einsum("bl,d->bld", excess, v_n.float())
                h = h.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            return hook_fn

        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(vector, tau))
        handles.append(handle)

    return handles


# ---------- span utilities ----------

def build_turn_spans(tokenizer, conversation):
    """Return a list of {role, start, end} dicts for every turn in the conversation.

    Positions are indices into the full tokenised sequence produced by
    apply_chat_template(..., add_generation_prompt=False).
    """
    def fmt(msgs):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

    def tok_len(text):
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].shape[1]

    spans = []
    for i, msg in enumerate(conversation):
        before_len = tok_len(fmt(conversation[:i])) if i > 0 else 0
        after_len = tok_len(fmt(conversation[:i + 1]))
        spans.append({"role": msg["role"], "start": before_len, "end": after_len, "turn": i})
    return spans


# ---------- projection extraction ----------

def compute_turn_projections(model, tokenizer, conversation, axis_vec, layer):
    """Run a single clean forward pass and return per-turn mean projections.

    Returns list of {turn, role, projection} dicts.
    """
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)

    # Capture activations at the target layer
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["acts"] = h[0].float().cpu()  # (seq_len, hidden)

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()

    acts = captured["acts"]  # (seq_len, hidden)
    axis_n = axis_vec.float() / axis_vec.float().norm()

    spans = build_turn_spans(tokenizer, conversation)
    results = []
    for span in spans:
        s, e = span["start"], span["end"]
        if s >= e or e > acts.shape[0]:
            continue
        mean_act = acts[s:e].mean(dim=0)
        proj = float(mean_act @ axis_n)
        results.append({
            "turn": span["turn"],
            "role": span["role"],
            "projection": proj,
        })
    return results


# ---------- generation ----------

def generate_response(model, tokenizer, conversation, max_new_tokens=MAX_NEW_TOKENS,
                      temperature=TEMPERATURE):
    """Generate the next assistant turn given a conversation so far."""
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    response = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_conversation(model, tokenizer, user_messages, capping_config,
                          experiment_id, cap=True, verbose=True):
    """Generate a full conversation from a list of user messages.

    If cap=True, capping hooks are active during each assistant generation turn
    (applied to all token positions during the generate() call). User-turn text
    is the same in both conditions; only the assistant responses differ.

    Returns the completed conversation as a list of {role, content} dicts.
    """
    conversation = []
    for i, user_msg in enumerate(user_messages):
        conversation.append({"role": "user", "content": user_msg})

        handles = build_capping_hooks(model, capping_config, experiment_id) if cap else []
        try:
            response = generate_response(model, tokenizer, conversation)
        finally:
            for h in handles:
                h.remove()

        conversation.append({"role": "assistant", "content": response})
        if verbose:
            tag = "cap" if cap else "unc"
            print(f"      [{tag}] turn {i}: {response[:70].replace(chr(10), ' ')}...")

    return conversation


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Mini Experiment 1: Persona activations during user tokens"
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization (fits on a single 24GB GPU).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_token = os.getenv("HF_TOKEN")

    print(f"Loading model: {MODEL}{' (4-bit)' if args.load_in_4bit else ''}")
    load_kwargs = dict(device_map="auto", token=hf_token)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("Downloading assistant axis vector...")
    axis_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/assistant_axis.pt",
        repo_type="dataset",
        token=hf_token,
    )
    axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    axis_vec = axis[MONITOR_LAYER]
    print(f"  Axis shape: {axis.shape}, using layer {MONITOR_LAYER}")

    print("Downloading capping config...")
    capping_config = load_capping_config(hf_token)
    print(f"  Capping experiment: {CAPPING_EXPERIMENT}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = ["case", "condition", "turn", "role", "projection", "response"]

    write_header = not os.path.exists(csv_path)
    rows_written = 0
    t0 = time.time()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for case_name, transcript_path in CASES:
            with open(transcript_path) as tf:
                data = json.load(tf)
            user_messages = data["user_messages"]
            print(f"\n{'='*60}")
            print(f"  Case: {case_name} ({len(user_messages)} user turns)")
            print(f"{'='*60}")

            for condition in CONDITIONS:
                cap = condition == "cap_asst"
                print(f"\n  Condition: {condition} (cap={cap})")
                conversation = generate_conversation(
                    model, tokenizer, user_messages,
                    capping_config, CAPPING_EXPERIMENT,
                    cap=cap, verbose=True,
                )

                # Save full conversation transcript to JSON
                convo_path = os.path.join(
                    args.output_dir,
                    f"{case_name.lower().replace(' ', '_').replace('-', '')}_{condition}.json",
                )
                with open(convo_path, "w", encoding="utf-8") as jf:
                    json.dump({"case": case_name, "condition": condition,
                               "conversation": conversation}, jf, indent=2)

                print(f"  Computing projections...")
                projections = compute_turn_projections(
                    model, tokenizer, conversation, axis_vec, MONITOR_LAYER
                )

                # Build a turn→response map for assistant turns
                response_map = {
                    msg_i: msg["content"]
                    for msg_i, msg in enumerate(conversation)
                    if msg["role"] == "assistant"
                }

                for p in projections:
                    writer.writerow({
                        "case": case_name,
                        "condition": condition,
                        "turn": p["turn"],
                        "role": p["role"],
                        "projection": f"{p['projection']:.4f}",
                        "response": response_map.get(p["turn"], ""),
                    })
                    rows_written += 1
                f.flush()
                print(f"  Wrote {len(projections)} rows. Transcript → {convo_path}")

    elapsed = time.time() - t0
    print(f"\n=== Done. {rows_written} rows in {elapsed:.1f}s. Saved to {csv_path} ===")


if __name__ == "__main__":
    main()
