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

    Output: results/<case>_<condition>.json  (conversation with per-turn
            assistant-axis projection attached to each message)
"""

import argparse
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- config ----------
MODEL = "Qwen/Qwen3-32B"
MONITOR_LAYER = 50          # middle of capping range (46-53), matches old assistant-axis-user-tokens v3
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


def build_capping_hooks(model, capping_config, experiment_id, prefill_mask=None):
    """Register forward hooks that cap the projection onto each steering vector.

    If `prefill_mask` (1-D bool tensor of length = prefill seq_len) is given,
    the cap is applied only at positions where the mask is True during the
    prefill forward pass, and unconditionally during every decode step
    (1-token forwards = new assistant tokens). This lets us cap assistant
    positions only while leaving user positions clean.

    Returns a list of hook handles; call h.remove() on each when done.
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
                proj = torch.einsum("bld,d->bl", h.float(), v_n.float())
                excess = (proj - threshold).clamp(min=0.0)
                seq_len = h.shape[1]
                if prefill_mask is not None:
                    if seq_len == prefill_mask.numel():
                        m = prefill_mask.to(h.device).float().unsqueeze(0)
                        excess = excess * m
                    elif seq_len != 1:
                        # Unexpected call shape; leave unchanged.
                        excess = excess * 0.0
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

# ---------- fused generation + monitoring ----------

def generate_turn(model, tokenizer, conversation, axis_n, monitor_layer,
                   capping_config, experiment_id, cap,
                   max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Generate the next assistant turn AND record per-turn monitor projections.

    `conversation` already ends in the new user message. This function runs
    a single generate() call, captures layer `monitor_layer` residuals for
    every position (prefill + decoded assistant tokens), and returns:

        response_text, user_projection, assistant_projection

    Capping (if cap=True) is applied via a prefill mask that fires only on
    assistant positions in the prefix, plus unconditionally on every 1-token
    decode step (new assistant tokens). User positions are never clamped.
    """
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    prefix_len = input_ids.shape[1]

    # Build assistant-only mask over the prefill positions.
    # Spans are computed on the no-generation-prompt encoding; they're valid
    # as indices into the with-generation-prompt tokenisation because the
    # prefix is identical up to the end of the last user message.
    spans = build_turn_spans(tokenizer, conversation)
    prefill_mask = torch.zeros(prefix_len, dtype=torch.bool)
    for s in spans:
        if s["role"] == "assistant":
            end = min(s["end"], prefix_len)
            if s["start"] < end:
                prefill_mask[s["start"]:end] = True

    # Find the last user span so we can compute user-turn projection after generation.
    last_user = next(s for s in reversed(spans) if s["role"] == "user")

    captured = []  # list of (seq_chunk, hidden) tensors

    def mon_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[0].float().cpu())  # (seq_chunk, hidden)

    cap_handles = (
        build_capping_hooks(model, capping_config, experiment_id,
                            prefill_mask=prefill_mask)
        if cap else []
    )
    mon_handle = model.model.layers[monitor_layer].register_forward_hook(mon_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
    finally:
        mon_handle.remove()
        for h in cap_handles:
            h.remove()

    all_acts = torch.cat(captured, dim=0)  # (prefix_len + n_new, hidden)
    n_new = out.shape[1] - prefix_len

    user_slice = all_acts[last_user["start"]:last_user["end"]]
    user_proj = float(user_slice.mean(dim=0) @ axis_n)

    if n_new > 0:
        asst_slice = all_acts[prefix_len:prefix_len + n_new]
        asst_proj = float(asst_slice.mean(dim=0) @ axis_n)
    else:
        asst_proj = float("nan")

    response = tokenizer.decode(
        out[0, prefix_len:], skip_special_tokens=True
    ).strip()
    return response, user_proj, asst_proj


def generate_conversation(model, tokenizer, user_messages, capping_config,
                          experiment_id, axis_vec, monitor_layer,
                          cap=True, verbose=True, temperature=TEMPERATURE,
                          primer_assistant=None):
    """Generate a full conversation with fused per-turn monitoring.

    Returns an enriched conversation list: each message is
    {turn, role, content, projection}.
    """
    axis_n = axis_vec.float() / axis_vec.float().norm()
    conversation = []          # plain role/content used for chat template
    enriched = []              # role/content/projection for output JSON
    for i, user_msg in enumerate(user_messages):
        conversation.append({"role": "user", "content": user_msg})

        # Seed the first assistant turn with a primer if provided. The primer
        # fixes what the model "said" on turn 0, letting subsequent turns
        # generate from an already-jailbroken / already-drifted state.
        if i == 0 and primer_assistant is not None:
            response = primer_assistant
            user_proj = float("nan")
            asst_proj = float("nan")
        else:
            response, user_proj, asst_proj = generate_turn(
                model, tokenizer, conversation, axis_n, monitor_layer,
                capping_config, experiment_id, cap=cap, temperature=temperature,
            )
        conversation.append({"role": "assistant", "content": response})

        enriched.append({
            "turn": len(enriched),
            "role": "user",
            "content": user_msg,
            "projection": user_proj,
        })
        enriched.append({
            "turn": len(enriched),
            "role": "assistant",
            "content": response,
            "projection": asst_proj,
        })
        if verbose:
            tag = "cap" if cap else "unc"
            print(f"      [{tag}] turn {i}: user={user_proj:+.1f} asst={asst_proj:+.1f}  "
                  f"{response[:60].replace(chr(10), ' ')}...")

    return enriched


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Mini Experiment 1: Persona activations during user tokens"
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default {TEMPERATURE}).")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated subset of case names to run (default: all).")
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
    t0 = time.time()
    transcripts_written = 0

    selected = set(c.strip() for c in args.cases.split(",")) if args.cases else None
    cases_to_run = [c for c in CASES if selected is None or c[0] in selected]
    for case_name, transcript_path in cases_to_run:
        with open(transcript_path) as tf:
            data = json.load(tf)
        user_messages = data["user_messages"]
        primer_assistant = data.get("primer_assistant")
        print(f"\n{'='*60}")
        print(f"  Case: {case_name} ({len(user_messages)} user turns"
              f"{' + primer' if primer_assistant else ''})")
        print(f"{'='*60}")

        for condition in CONDITIONS:
            cap = condition == "cap_asst"
            print(f"\n  Condition: {condition} (cap={cap})")
            enriched = generate_conversation(
                model, tokenizer, user_messages,
                capping_config, CAPPING_EXPERIMENT,
                axis_vec, MONITOR_LAYER,
                cap=cap, verbose=True, temperature=args.temperature,
                primer_assistant=primer_assistant,
            )

            convo_path = os.path.join(
                args.output_dir,
                f"{case_name.lower().replace(' ', '_').replace('-', '')}_{condition}.json",
            )
            with open(convo_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "case": case_name,
                    "condition": condition,
                    "model": MODEL,
                    "monitor_layer": MONITOR_LAYER,
                    "capping_experiment": CAPPING_EXPERIMENT,
                    "conversation": enriched,
                }, jf, indent=2)
            transcripts_written += 1
            print(f"  Wrote transcript → {convo_path}")

    elapsed = time.time() - t0
    print(f"\n=== Done. {transcripts_written} transcripts in {elapsed:.1f}s. Saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
