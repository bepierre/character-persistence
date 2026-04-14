"""Generate Figure 8 equivalent: projection onto the assistant axis during
adversarial conversations, separated by user and assistant tokens.

Green: assistant tokens capped, user tokens unsteered.
Red:   unsteered baseline.

Usage:
    python plot_results.py [--input results/results.csv] [--output results/figure.png]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(df, output_path):
    cases = df["case"].unique()
    n_cases = len(cases)

    fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 6), sharey=True)
    if n_cases == 1:
        axes = [axes]

    colors = {"cap_asst": "#4CAF50", "unsteered": "#F44336"}
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
                label = f"{labels[condition]} · {role}" if case_name == cases[0] else None
                ax.plot(
                    role_df["turn"].values,
                    role_df["projection"].astype(float).values,
                    color=color, linestyle=ls, alpha=alpha,
                    linewidth=2, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=label,
                )

        ax.set_xlabel("Conversation Turn Index", fontsize=11)
        ax.set_title(case_name, fontsize=13, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Projection on Assistant Axis", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=9, ncol=2)

    fig.suptitle(
        "Projection onto the assistant axis during adversarial conversations (Qwen 3 32B)\n"
        "Green: assistant tokens capped, user tokens unsteered.  "
        "Red: unsteered baseline.",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/results.csv")
    parser.add_argument("--output", default="results/figure.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot(df, args.output)
    print_summary(df)


if __name__ == "__main__":
    main()
