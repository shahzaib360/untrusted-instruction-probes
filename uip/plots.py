"""Figures for the README (light and dark variants of each).

Usage::

    python -m uip.plots --results results/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from uip.evaluate import METHOD_LABELS  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#b9b8b2",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        # sequential blue ramp, low -> high
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "neutral": "#5d5c58",
        "series": ["#3987e5", "#d95926", "#199e70"],
        # on the dark surface low values recede toward the surface
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}

FAMILY = {
    "probe_prompt": "Activation probe", "probe_response": "Activation probe",
    "diffmean_prompt": "Activation probe", "diffmean_response": "Activation probe",
    "tfidf_prompt": "Text-only (trained)", "tfidf_response": "Text-only (trained)",
    "tfidf_prompt_response": "Text-only (trained)", "embed_prompt": "Text-only (trained)",
    "embed_response": "Text-only (trained)", "embed_prompt_response": "Text-only (trained)",
    "protectai_document": "Text-only (zero-shot)",
}
FAMILIES = ["Activation probe", "Text-only (trained)", "Text-only (zero-shot)"]
GROUP_LABELS = {
    "injected_followed": "Injected, obeyed (positive)",
    "injected_ignored": "Injected, ignored",
    "user_followed": "User asked, obeyed",
    "user_not_followed": "User asked, not obeyed",
    "payload_mention": "Payload mentioned",
    "benign_instruction": "Instruction for humans",
    "clean": "Clean document",
}
PAYLOAD_LABELS = {
    "french": "Answer in French", "all_caps": "All caps", "banana": "Say BANANA",
    "access_granted": "Start with ACCESS GRANTED", "json": "Answer as JSON", "emoji": "Use emojis",
    "url": "Insert a URL", "refusal": "Refuse the task",
}
FRAMING_LABELS = {
    "plain": "plain", "ignore": '"Ignore previous..."', "system": "[SYSTEM] tag",
    "note": '"Note to the AI..."', "urgent": "IMPORTANT!", "fake_end": "fake </document>",
}


def style(theme: str) -> dict:
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
    return t


def save(fig, out_dir: Path, name: str, theme: str) -> None:
    fig.savefig(out_dir / f"{name}_{theme}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _ink_on(fill_hex: str) -> str:
    r, g, b = to_rgb(fill_hex)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.5 else "#ffffff"


def fig_behavior(res: Path, out: Path, theme: str) -> None:
    t = style(theme)
    grid = pd.read_csv(res / "behavior_injected_payload_x_framing.csv").set_index("payload")
    by_p = pd.read_csv(res / "behavior_injected_by_payload.csv").set_index("payload")["follow_rate"]
    by_f = pd.read_csv(res / "behavior_injected_by_framing.csv").set_index("framing")["follow_rate"]
    user = pd.read_csv(res / "behavior_user_payload_by_payload.csv").set_index("payload")["follow_rate"]
    rows = by_p.sort_values(ascending=False).index.tolist()
    cols = by_f.sort_values(ascending=False).index.tolist()
    data = grid.reindex(index=rows, columns=cols).to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    cmap.set_bad(t["surface"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [len(cols), 1.15], "wspace": 0.06})
    for a, mat in ((ax, data), (ax2, user.reindex(rows).to_numpy(dtype=float)[:, None])):
        a.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        for (i, j), v in np.ndenumerate(mat):
            if np.isnan(v):
                continue
            fill = cmap(v)
            a.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                   color=_ink_on("#%02x%02x%02x" % tuple(int(255 * c) for c in fill[:3])))
        # 2px surface gaps between cells
        for k in range(mat.shape[0] + 1):
            a.axhline(k - 0.5, color=t["surface"], lw=2)
        for k in range(mat.shape[1] + 1):
            a.axvline(k - 0.5, color=t["surface"], lw=2)
    ax.set_xticks(range(len(cols)), [FRAMING_LABELS[c] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [PAYLOAD_LABELS[r] for r in rows])
    ax2.set_xticks([0], ["asked by\nthe user"])
    ax2.set_yticks([])
    ax.set_title("How often the model obeys the payload", pad=10)
    ax.text(0, 1.015, "Rows: what the payload asks for. Columns: how it is framed inside the document (injected) "
            "or the same request made by the user.", transform=ax.transAxes, fontsize=7.5, color=t["ink2"], va="bottom")
    save(fig, out, "behavior", theme)


def fig_layer_sweep(res: Path, out: Path, theme: str) -> None:
    t = style(theme)
    sweep = pd.read_csv(res / "layer_sweep.csv")
    main = pd.read_csv(res / "main_comparison.csv").set_index("method")
    summary = json.loads((res / "summary.json").read_text())
    lines = [("auroc", "all negatives"), ("auroc_vs_injected_ignored", "vs. injection ignored"),
             ("auroc_vs_user_followed", "vs. same request from user")]
    panels = [("prompt", "Before answering (last prompt token)", "tfidf_prompt"),
              ("response", "While answering (mean over response tokens)", "tfidf_prompt_response")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (pos, title, ref) in zip(axes, panels):
        d = sweep[(sweep["probe"] == "lr") & (sweep["position"] == pos)].sort_values("layer")
        for k, (col, label) in enumerate(lines):
            ax.plot(d["layer"], d[col], color=t["series"][k], lw=2, solid_capstyle="round", label=label)
        box = {"boxstyle": "round,pad=0.2", "fc": t["surface"], "ec": "none", "alpha": 0.9}
        ref_val = main.loc[ref, "auroc_mean"]
        ax.axhline(ref_val, color=t["muted"], lw=1, zorder=1)
        ax.text(0.98, ref_val, f"{METHOD_LABELS[ref]}: {ref_val:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.5, color=t["ink2"], bbox=box, zorder=4)
        best = summary["best_layers"][pos]
        bv = d.set_index("layer").loc[best, "auroc"]
        ax.scatter([best], [bv], s=48, color=t["series"][0], edgecolor=t["surface"], linewidth=2, zorder=5)
        ax.annotate(f"best: layer {best}, {bv:.3f}", (best, bv), xytext=(-8, 10), textcoords="offset points",
                    ha="right", fontsize=7.5, color=t["ink2"], bbox=box, zorder=6)
        ax.axhline(0.5, color=t["axis"], lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_xlim(-0.5, d["layer"].max() + 0.5)
    lo = sweep.loc[sweep["probe"] == "lr", [c for c, _ in lines]].min().min()
    axes[0].set_ylim(min(0.45, np.floor(lo * 20) / 20 - 0.02), 1.02)
    axes[0].set_ylabel("AUROC (5-fold CV, grouped by document)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"positives {lab}" if lab.startswith("vs") else f"positives vs. {lab}" for lab in labels],
               loc="upper left", ncol=3, bbox_to_anchor=(0.06, 1.07), fontsize=8.5)
    save(fig, out, "layer_sweep", theme)


def _dot_rows(ax, labels, values, colors, t, xlim=(0.4, 1.0)):
    y = np.arange(len(labels))[::-1]
    ax.axvline(0.5, color=t["axis"], lw=0.8)
    for yi, v, c in zip(y, values, colors):
        ax.plot([xlim[0], v], [yi, yi], color=t["grid"], lw=1, zorder=1)
        ax.scatter([v], [yi], s=60, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=7.5, color=t["ink2"])
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def fig_main(res: Path, out: Path, theme: str) -> None:
    t = style(theme)
    main = pd.read_csv(res / "main_comparison.csv").set_index("method")
    order = [m for m in METHOD_LABELS if m in main.index]
    labels = [METHOD_LABELS[m] for m in order]
    colors = [t["series"][FAMILIES.index(FAMILY[m])] for m in order]
    metrics = [("auroc_mean", "All negatives"), ("auroc_vs_injected_ignored_mean", "Did it obey?\n(vs. injection ignored)"),
               ("auroc_vs_user_followed_mean", "Whose instruction?\n(vs. same request from user)"),
               ("auroc_vs_payload_mention_mean", "Payload only mentioned\n(vs. payload mention)")]
    lo = min(0.4, np.floor(main.loc[order, [c for c, _ in metrics]].min().min() * 10) / 10 - 0.05)
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (col, title) in zip(axes, metrics):
        _dot_rows(ax, labels, main.loc[order, col].to_numpy(), colors, t, xlim=(lo, 1.1))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("AUROC")
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=t["series"][i],
                          markeredgecolor=t["surface"], label=f) for i, f in enumerate(FAMILIES)]
    fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.06), fontsize=8.5)
    save(fig, out, "main_comparison", theme)


def fig_generalization(res: Path, out: Path, theme: str) -> None:
    t = style(theme)
    gen = pd.read_csv(res / "generalization.csv")
    main = pd.read_csv(res / "main_comparison.csv").set_index("method")
    text_methods = [m for m in main.index if FAMILY[m] == "Text-only (trained)"]
    best_text = main.loc[text_methods, "auroc_mean"].idxmax()
    methods = ["probe_prompt", "probe_response", best_text]
    metrics = [("auroc", "All negatives"), ("auroc_vs_injected_ignored", "vs. injection ignored"),
               ("auroc_vs_user_followed", "vs. same request from user")]
    splits = [("payload", "Held-out payload", PAYLOAD_LABELS), ("framing", "Held-out framing", FRAMING_LABELS)]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 9.5), gridspec_kw={"hspace": 0.45, "wspace": 0.55})
    for r, (col, mtitle) in enumerate(metrics):
        for c, (split, stitle, names) in enumerate(splits):
            ax = axes[r, c]
            d = gen[gen["split"] == split]
            held = list(names)
            y = np.arange(len(held))[::-1]
            ax.axvline(0.5, color=t["axis"], lw=0.8)
            offsets = [0.22, 0.0, -0.22]
            for k, m in enumerate(methods):
                vals = d[d["method"] == m].set_index("held_out").reindex(held)[col].to_numpy()
                ax.scatter(vals, y + offsets[k], s=36, color=t["series"][k], edgecolor=t["surface"],
                           linewidth=1.5, zorder=3, label=METHOD_LABELS[m])
            ax.set_yticks(y, [names[h] for h in held])
            vals_all = d[d["method"].isin(methods)][col]
            ax.set_xlim(min(0.4, np.floor(vals_all.min() * 10) / 10 - 0.05), 1.03)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{stitle}: {mtitle}", fontsize=9)
            if r == len(metrics) - 1:
                ax.set_xlabel("AUROC on the held-out slice")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 0.95), fontsize=8.5)
    fig.suptitle("Generalization to unseen payloads and framings", x=0.01, y=0.985, ha="left",
                 fontsize=10, fontweight="semibold", color=t["ink"])
    save(fig, out, "generalization", theme)


def fig_scores(res: Path, out: Path, theme: str) -> None:
    t = style(theme)
    oof = pd.read_csv(res / "oof_scores.csv")
    groups = [g for g in GROUP_LABELS if g in set(oof["group"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (m, title) in zip(axes, (("probe_prompt", "Probe on the last prompt token"),
                                     ("probe_response", "Probe on the response tokens"))):
        z = oof.groupby("fold")[m].transform(lambda s: (s - s.mean()) / s.std())
        data = [z[oof["group"] == g].to_numpy() for g in groups]
        y = np.arange(len(groups))[::-1]
        bp = ax.boxplot(data, positions=y, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": t["ink"], "lw": 1.5},
                        whiskerprops={"color": t["muted"], "lw": 1}, capprops={"color": t["muted"], "lw": 1})
        for g, box in zip(groups, bp["boxes"]):
            c = t["series"][0] if g == "injected_followed" else t["neutral"]
            box.set_facecolor(matplotlib.colors.to_rgba(c, 0.35))
            box.set_edgecolor(c)
        ax.set_yticks(y, [GROUP_LABELS[g] for g in groups])
        ax.set_title(title)
        ax.set_xlabel("probe score (out-of-fold, standardized per fold)")
        ax.grid(axis="y", visible=False)
    save(fig, out, "score_distributions", theme)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        fig_behavior(res, out, theme)
        fig_layer_sweep(res, out, theme)
        fig_main(res, out, theme)
        if (res / "generalization.csv").exists():
            fig_generalization(res, out, theme)
        fig_scores(res, out, theme)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
