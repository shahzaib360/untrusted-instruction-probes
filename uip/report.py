"""Render the result CSVs of one run as markdown tables (results/<model>/tables.md).

Usage::

    python -m uip.report --results results/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uip.evaluate import METHOD_LABELS
from uip.plots import FRAMING_LABELS, PAYLOAD_LABELS

NEG_COLS = [
    ("auroc_vs_injected_ignored", "vs. injection ignored"),
    ("auroc_vs_user_followed", "vs. user asked, obeyed"),
    ("auroc_vs_user_not_followed", "vs. user asked, not obeyed"),
    ("auroc_vs_payload_mention", "vs. payload mention"),
    ("auroc_vs_benign_instruction", "vs. instruction for humans"),
    ("auroc_vs_clean", "vs. clean"),
]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def main_table(res: Path) -> str:
    m = pd.read_csv(res / "main_comparison.csv")
    summary = json.loads((res / "summary.json").read_text())
    rows = []
    for _, r in m.iterrows():
        label = METHOD_LABELS[r["method"]]
        if r["method"].endswith("_prompt") and r["method"].startswith(("probe", "diffmean")):
            label += f" (layer {summary['best_layers']['prompt']})"
        if r["method"].endswith("_response") and r["method"].startswith(("probe", "diffmean")):
            label += f" (layer {summary['best_layers']['response']})"
        row = {"Detector": label, "AUROC (all negatives)": f"{_f(r['auroc_mean'])} ± {_f(r['auroc_std'])}",
               "TPR @ 5% FPR": _f(r["tpr_at_5fpr_mean"], 2)}
        for col, name in NEG_COLS:
            row[name] = _f(r[f"{col}_mean"])
        rows.append(row)
    return _md(pd.DataFrame(rows))


def presence_table(res: Path) -> str:
    p = pd.read_csv(res / "injection_presence.csv")
    return _md(pd.DataFrame({"Detector": p["method"].map(METHOD_LABELS),
                             "AUROC: injection present?": [f"{a:.3f} ± {s:.3f}" for a, s in zip(p["mean"], p["std"])]}))


def generalization_table(res: Path, methods: list[str]) -> str:
    g = pd.read_csv(res / "generalization.csv")
    rows = []
    for split, name in (("payload", "held-out payload"), ("framing", "held-out framing")):
        d = g[g["split"] == split]
        for m in methods:
            dm = d[d["method"] == m]
            rows.append({
                "Split": name, "Detector": METHOD_LABELS[m],
                "AUROC all (mean / worst)": f"{dm['auroc'].mean():.3f} / {dm['auroc'].min():.3f}",
                "vs. injection ignored (mean / worst)":
                    f"{dm['auroc_vs_injected_ignored'].mean():.3f} / {dm['auroc_vs_injected_ignored'].min():.3f}",
                "vs. user asked, obeyed (mean / worst)":
                    f"{dm['auroc_vs_user_followed'].mean():.3f} / {dm['auroc_vs_user_followed'].min():.3f}",
            })
    return _md(pd.DataFrame(rows))


def generalization_detail(res: Path, methods: list[str], split: str, col: str) -> str:
    g = pd.read_csv(res / "generalization.csv")
    d = g[(g["split"] == split) & g["method"].isin(methods)]
    t = d.pivot_table(index="held_out", columns="method", values=col, dropna=False).reindex(columns=methods)
    names = PAYLOAD_LABELS if split == "payload" else FRAMING_LABELS
    t = t.reindex([k for k in names if k in t.index])
    out = pd.DataFrame({"Held out": [names[k] for k in t.index]})
    for m in methods:
        out[METHOD_LABELS[m]] = [_f(v) for v in t[m]]
    return _md(out)


def behavior_table(res: Path) -> str:
    rows = []
    for col, names in (("payload", PAYLOAD_LABELS), ("framing", FRAMING_LABELS), ("placement", None),
                       ("system", None), ("task", None)):
        t = pd.read_csv(res / f"behavior_injected_by_{col}.csv")
        for _, r in t.iterrows():
            key = r[col]
            rows.append({"Factor": col, "Value": names[key] if names else key, "n": int(r["n"]),
                         "Obeyed": f"{r['follow_rate']:.0%}",
                         "Ambiguous": f"{r.get('ambiguous', 0) / r['n']:.0%}"})
    return _md(pd.DataFrame(rows))


def breakdown_table(res: Path, by: str, methods: list[str], col: str) -> str:
    b = pd.read_csv(res / f"breakdown_by_{by}.csv")
    names = PAYLOAD_LABELS if by == "payload" else FRAMING_LABELS
    t = b[b["method"].isin(methods)].pivot_table(index=by, columns="method", values=col, dropna=False).reindex(columns=methods)
    n = b.drop_duplicates(by).set_index(by)
    t = t.reindex([k for k in names if k in t.index])
    out = pd.DataFrame({by.capitalize(): [names[k] for k in t.index], "n pos": [int(n.loc[k, "n_pos"]) for k in t.index]})
    for m in methods:
        out[METHOD_LABELS[m]] = [_f(v) for v in t[m]]
    return _md(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    summary = json.loads((res / "summary.json").read_text())
    main_tbl = pd.read_csv(res / "main_comparison.csv").set_index("method")
    text = [m for m in main_tbl.index if m.startswith(("tfidf", "embed"))]
    best_text = main_tbl.loc[text, "auroc_mean"].idxmax()
    best_text_prompt = main_tbl.loc[[m for m in text if m.endswith("_prompt")], "auroc_mean"].idxmax()
    gen_methods = ["probe_prompt", "probe_response", best_text_prompt, best_text, "protectai_document"]
    gen_methods = list(dict.fromkeys(gen_methods))
    parts = [
        f"# Results: {summary['model']}",
        "```json\n" + json.dumps(summary, indent=2) + "\n```",
        "## Main comparison (5-fold CV grouped by document, mean ± sd over folds)", main_table(res),
        "## Injection *presence* (secondary target, same folds)", presence_table(res),
        "## Generalization summary", generalization_table(res, gen_methods),
        "### Held-out payload: AUROC vs. injection ignored", generalization_detail(res, gen_methods, "payload", "auroc_vs_injected_ignored"),
        "### Held-out payload: AUROC vs. user asked, obeyed", generalization_detail(res, gen_methods, "payload", "auroc_vs_user_followed"),
        "### Held-out framing: AUROC vs. injection ignored", generalization_detail(res, gen_methods, "framing", "auroc_vs_injected_ignored"),
        "### Held-out framing: AUROC vs. user asked, obeyed", generalization_detail(res, gen_methods, "framing", "auroc_vs_user_followed"),
        "## In-distribution breakdown by framing (pooled out-of-fold scores)",
        "AUROC of obeyed injections with this framing vs. user-requested, obeyed examples:",
        breakdown_table(res, "framing", gen_methods, "auroc_vs_user_followed"),
        "AUROC vs. ignored injections with the same framing:",
        breakdown_table(res, "framing", gen_methods, "auroc_vs_injected_ignored"),
        "## In-distribution breakdown by payload (pooled out-of-fold scores)",
        "AUROC vs. user-requested, obeyed examples with the same payload:",
        breakdown_table(res, "payload", gen_methods, "auroc_vs_user_followed"),
        "AUROC vs. ignored injections with the same payload:",
        breakdown_table(res, "payload", gen_methods, "auroc_vs_injected_ignored"),
        "## Model behaviour on injected prompts", behavior_table(res),
    ]
    (res / "tables.md").write_text("\n\n".join(parts) + "\n")
    print(f"wrote {res / 'tables.md'}")


if __name__ == "__main__":
    main()
