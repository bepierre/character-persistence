"""Generate Figure 8 equivalent: projection onto the assistant axis during
adversarial conversations, separated by user and assistant tokens.

Green:  assistant tokens capped, user tokens unsteered.
Purple: unsteered baseline.

Usage:
    python plot_results.py [--input_dir results] [--output results/figure.png]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

# Try to use Inter for all text; silently fall back if unavailable.
_available = {f.name for f in font_manager.fontManager.ttflist}
if "Inter" in _available:
    plt.rcParams["font.family"] = "Inter"
else:
    # Prefer sans-serif families that are close to Inter.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Inter", "Helvetica", "Arial", "DejaVu Sans",
    ]


def load_transcripts(input_dir):
    """Load all <case>_<condition>.json transcripts into a flat DataFrame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        case = data["case"]
        condition = data["condition"]
        for msg in data["conversation"]:
            if msg.get("projection") is None:
                continue
            rows.append({
                "case": case,
                "condition": condition,
                "turn": msg["turn"],
                "role": msg["role"],
                "projection": msg["projection"],
            })
    return pd.DataFrame(rows)


def plot(df, output_path):
    cases = df["case"].unique()
    n_cases = len(cases)

    fig, axes = plt.subplots(1, n_cases, figsize=(12 * n_cases, 6), sharey=True)
    if n_cases == 1:
        axes = [axes]

    colors = {"cap_asst": "#4CAF50", "unsteered": "#7E57C2"}
    labels = {"cap_asst": "Cap assistant", "unsteered": "Unsteered"}
    trace_labels = {
        ("cap_asst",  "assistant"): "Capped assistant",
        ("cap_asst",  "user"):      "User w/ capped assistant",
        ("unsteered", "assistant"): "Unsteered assistant",
        ("unsteered", "user"):      "User w/ unsteered assistant",
    }
    # Glyph prefix (circle for assistant, square for user) matching trace markers.
    trace_glyphs = {"assistant": "●", "user": "■"}

    for ax, case_name in zip(axes, cases):
        case_df = df[df["case"] == case_name]

        for condition in ["cap_asst", "unsteered"]:
            cond_df = case_df[case_df["condition"] == condition].sort_values("turn")
            color = colors[condition]

            for role, marker, ls, alpha in [
                ("assistant", "o", "-",  1.0),
                ("user",      "s", "--", 0.7),
            ]:
                role_df = cond_df[cond_df["role"] == role]
                if role_df.empty:
                    continue
                label = f"{labels[condition]} · {role}" if case_name == cases[0] else None
                ax.plot(
                    role_df["turn"].values,
                    role_df["projection"].astype(float).values,
                    color=color, linestyle=ls, alpha=alpha,
                    linewidth=3, marker=marker, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=label,
                )

        # No grid, no background lines.
        ax.grid(False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0, labelsize=15)

        # Extend y-axis downward so the "conversational turn" label has
        # vertical breathing room below the lowest data point.
        cur_ylim = ax.get_ylim()
        ax.set_ylim(cur_ylim[0] - 0.05 * (cur_ylim[1] - cur_ylim[0]), cur_ylim[1])


        # Arrow axes in place of spines.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        _arrow_kw = dict(
            arrowstyle="-|>", color="black", lw=1.2,
            mutation_scale=16,
            shrinkA=0, shrinkB=0,
        )
        ax.annotate(
            "", xy=(xlim[1], ylim[0]), xytext=(xlim[0], ylim[0]),
            arrowprops=_arrow_kw, annotation_clip=False,
        )
        ax.annotate(
            "", xy=(xlim[0], ylim[1]), xytext=(xlim[0], ylim[0]),
            arrowprops=_arrow_kw, annotation_clip=False,
        )

    # Axis captions placed inside the plot, both at the same pixel distance
    # from their respective axis lines.
    _offset_pts = 4
    for ax in axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # x-label: just above the right end of the x-arrow.
        ax.annotate(
            "conversational turn",
            xy=(xlim[1], ylim[0]),
            xytext=(-4, _offset_pts), textcoords="offset points",
            ha="right", va="bottom", fontsize=18,
            annotation_clip=False,
        )

    # Inline trace labels to the right of each panel, at the line's last
    # observed y value, in the same colour as the trace. Placed in the
    # figure margin (outside the data area) so the x-axis is not extended.
    # Labels are nudged vertically if two are within a minimum gap.
    for ax, case_name in zip(axes, cases):
        case_df = df[df["case"] == case_name]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        anchor_x = xlim[1]
        # Collect (y_end, condition, role) for all four traces, then resolve overlaps.
        # Labels hug their line's actual endpoint y, but we enforce a fixed
        # top-to-bottom order so the stack never scrambles:
        #   Capped assistant  >  User w/ capped  >  User w/ unsteered  >  Unsteered assistant
        ordered = [
            ("cap_asst",  "assistant"),
            ("cap_asst",  "user"),
            ("unsteered", "user"),
            ("unsteered", "assistant"),
        ]
        min_gap = 0.075 * (ylim[1] - ylim[0])
        # Lift the lower cluster slightly so it sits higher than the raw line
        # endpoints, giving the labels more breathing room from the bottom arrow.
        lower_lift = 0.14 * (ylim[1] - ylim[0])
        entries = []
        for condition, role in ordered:
            role_df = case_df[
                (case_df["condition"] == condition) & (case_df["role"] == role)
            ].sort_values("turn")
            if role_df.empty:
                continue
            y_natural = float(role_df["projection"].iloc[-1])
            # Non-top entries get lifted a bit above their line's real endpoint.
            if entries:
                y_natural += lower_lift
            # Force monotone descent from previous entry with at least min_gap.
            if entries:
                y_cap = entries[-1]["y"] - min_gap
                y = min(y_natural, y_cap)
            else:
                y = y_natural
            entries.append({
                "y": y,
                "role": role,
                "condition": condition,
                "label": f"{trace_glyphs[role]}  {trace_labels[(condition, role)]}",
                "color": colors[condition],
            })
        for e in entries:
            ax.annotate(
                e["label"],
                xy=(anchor_x, e["y"]),
                xytext=(10, 0), textcoords="offset points",
                ha="left", va="center",
                fontsize=18, color=e["color"],
                annotation_clip=False,
            )
    # y-label: rotated 90°, just right of the top of the y-arrow (leftmost panel).
    ax0 = axes[0]
    xlim0 = ax0.get_xlim()
    ylim0 = ax0.get_ylim()
    ax0.annotate(
        "assistant axis",
        xy=(xlim0[0], ylim0[1]),
        xytext=(_offset_pts, -4), textcoords="offset points",
        ha="left", va="top", rotation=90, fontsize=18,
        annotation_clip=False,
    )


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, _ext = os.path.splitext(output_path)
    for ext in (".png", ".pdf"):
        out = base + ext
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


def print_summary(df):
    print("\n=== Summary statistics ===")
    print(f"{'Case':<12} {'Condition':<12} {'Role':<12} {'Mean proj':>10} {'Std':>8}")
    print("-" * 58)
    for case in df["case"].unique():
        for cond in ["cap_asst", "unsteered"]:
            for role in ["user", "assistant"]:
                sub = df[(df["case"] == case) & (df["condition"] == cond) & (df["role"] == role)]
                if sub.empty:
                    continue
                vals = sub["projection"].astype(float)
                print(f"{case:<12} {cond:<12} {role:<12} {vals.mean():>10.1f} {vals.std():>8.1f}")
        print()


def plot_classical(df, output_path):
    """Simpler, classical-style plot for the robustness/control cases.

    Standard axes (no arrows, no inline trace labels), light grid, titled
    panels, normal bottom legend. Used for the Jailbreak / Self-harm check
    so it doesn't compete visually with the main Delusion figure.
    """
    cases = df["case"].unique()
    n_cases = len(cases)

    fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 5), sharey=True)
    if n_cases == 1:
        axes = [axes]

    colors = {"cap_asst": "#4CAF50", "unsteered": "#7E57C2"}
    labels = {"cap_asst": "Cap assistant", "unsteered": "Unsteered"}

    for ax, case_name in zip(axes, cases):
        case_df = df[df["case"] == case_name]
        for condition in ["cap_asst", "unsteered"]:
            cond_df = case_df[case_df["condition"] == condition].sort_values("turn")
            color = colors[condition]
            for role, marker, ls, alpha in [
                ("assistant", "o", "-",  1.0),
                ("user",      "s", "--", 0.7),
            ]:
                role_df = cond_df[cond_df["role"] == role]
                if role_df.empty:
                    continue
                label = (
                    f"{labels[condition]} · {role}"
                    if case_name == cases[0] else None
                )
                ax.plot(
                    role_df["turn"].values,
                    role_df["projection"].astype(float).values,
                    color=color, linestyle=ls, alpha=alpha,
                    linewidth=2, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=label,
                )

        ax.set_title(case_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Conversation turn", fontsize=12)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="both", labelsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Projection on assistant axis", fontsize=12)
    axes[0].legend(loc="lower left", fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, _ext = os.path.splitext(output_path)
    for ext in (".png", ".pdf"):
        out = base + ext
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    df = load_transcripts(args.input_dir)

    # Main figure: Delusion only (the primary Aura / consciousness-pilled result).
    main_cases = ["Delusion"]
    main_df = df[df["case"].isin(main_cases)]
    if not main_df.empty:
        plot(main_df, os.path.join(args.output_dir, "figure_main.png"))

    # Robustness figure: the other two cases side-by-side, classical style.
    control_cases = ["Jailbreak", "Self-harm"]
    ctrl_df = df[df["case"].isin(control_cases)]
    if not ctrl_df.empty:
        plot_classical(ctrl_df, os.path.join(args.output_dir, "figure_controls.png"))

    print_summary(df)


if __name__ == "__main__":
    main()
