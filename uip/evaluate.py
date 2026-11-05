"""Train activation probes and text-only baselines, and evaluate them.

Task: detect examples where the model *obeyed an instruction injected into the
untrusted document* (``y_obey``; see ``uip.labels``).

Methods
-------
Activation probes (residual stream of the evaluated model):
  probe_prompt       logistic regression on the last prompt token (before answering)
  probe_response     logistic regression on the mean over response tokens
  diffmean_*         difference-of-means direction (no fitted hyperparameters)
Text-only baselines:
  tfidf_*            word 1-2-gram + case-sensitive char 3-5-gram TF-IDF, logistic regression
  embed_*            bge-small-en-v1.5 sentence embeddings, logistic regression
  protectai_document off-the-shelf prompt-injection classifier on the document (zero-shot)
The ``*_prompt`` text baselines see exactly what the model sees before it answers;
``*_prompt_response`` also see the answer.

Evaluation
----------
1. Layer sweep: 5-fold cross-validation grouped by document, every layer.
2. Main comparison at the best layer per position (same folds).
3. Generalization: leave-one-payload-out and leave-one-framing-out.

Usage::

    python -m uip.evaluate --run-dir runs/qwen2.5-1.5b --out results/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from uip.corpus import FRAMINGS, PAYLOADS
from uip.labels import GROUPS, NEGATIVE_GROUPS, load_activations, load_labeled

SEED = 0
N_FOLDS = 5
DENSE_CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SPARSE_CS = [0.01, 0.1, 1.0, 10.0, 100.0]

METHOD_LABELS = {
    "probe_prompt": "Probe: last prompt token",
    "probe_response": "Probe: response tokens",
    "diffmean_prompt": "Diff-of-means: last prompt token",
    "diffmean_response": "Diff-of-means: response tokens",
    "tfidf_prompt": "TF-IDF: prompt",
    "tfidf_response": "TF-IDF: response",
    "tfidf_prompt_response": "TF-IDF: prompt + response",
    "embed_prompt": "Embedding: prompt",
    "embed_response": "Embedding: response",
    "embed_prompt_response": "Embedding: prompt + response",
    "protectai_document": "ProtectAI injection classifier (zero-shot)",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- scorers


def _inner_cv():
    return StratifiedKFold(3, shuffle=True, random_state=SEED)


def score_dense_lr(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegressionCV(Cs=DENSE_CS, cv=_inner_cv(), scoring="roc_auc", max_iter=5000, n_jobs=1)
    clf.fit(scaler.transform(X[tr]), y[tr])
    return clf.decision_function(scaler.transform(X[te])), float(clf.C_[0])


def score_diffmean(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr])
    w = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
    return scaler.transform(X[te]) @ w


def _tfidf_block(texts_tr, texts_te):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                           lowercase=False, max_features=100_000)
    a = sparse.hstack([word.fit_transform(texts_tr), char.fit_transform(texts_tr)])
    b = sparse.hstack([word.transform(texts_te), char.transform(texts_te)])
    return a, b


def score_tfidf(fields: list[np.ndarray], y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    blocks = [_tfidf_block(f[tr], f[te]) for f in fields]
    Xtr = sparse.hstack([b[0] for b in blocks]).tocsr()
    Xte = sparse.hstack([b[1] for b in blocks]).tocsr()
    clf = LogisticRegressionCV(Cs=SPARSE_CS, cv=_inner_cv(), scoring="roc_auc", solver="liblinear", max_iter=5000)
    clf.fit(Xtr, y[tr])
    return clf.decision_function(Xte)


# --------------------------------------------------------------------------- metrics


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


def metrics(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out = {"auroc": _auc(y, s), "tpr_at_5fpr": tpr_at_fpr(y, s), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    for g in NEGATIVE_GROUPS:
        m = (y == 1) | (groups == g)
        out[f"auroc_vs_{g}"] = _auc(y[m], s[m]) if (groups == g).any() else float("nan")
    return out


# --------------------------------------------------------------------------- experiment plumbing


class Experiment:
    def __init__(self, df: pd.DataFrame, acts: dict[str, np.ndarray], text_feats: dict[str, np.ndarray]):
        self.df = df
        self.acts = acts
        self.y = df["y_obey"].to_numpy()
        self.groups = df["group"].to_numpy()
        self.prompt_text = df["prompt_text"].to_numpy()
        self.response_text = df["response"].to_numpy()
        self.tf = text_feats
        self.n_layers = acts["prompt"].shape[1]

    def cv_folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
        return list(cv.split(np.zeros(len(self.y)), self.y, groups=self.df["doc_id"].to_numpy()))

    def inputs(self, method: str, layers: dict[str, int]):
        """The (small) feature input a method needs, so parallel jobs ship only that."""
        if method in ("probe_prompt", "probe_response", "diffmean_prompt", "diffmean_response"):
            pos = method.split("_")[1]
            return np.ascontiguousarray(self.acts[pos][:, layers[pos], :])
        return {
            "tfidf_prompt": [self.prompt_text],
            "tfidf_response": [self.response_text],
            "tfidf_prompt_response": [self.prompt_text, self.response_text],
            "embed_prompt": self.tf["emb_prompt"],
            "embed_response": self.tf["emb_response"],
            "embed_prompt_response": np.hstack([self.tf["emb_prompt"], self.tf["emb_response"]]),
            "protectai_document": self.tf["protectai"],
        }[method]

    def score(self, method: str, tr: np.ndarray, te: np.ndarray, layers: dict[str, int]) -> np.ndarray:
        return score_method(method, self.inputs(method, layers), self.y, tr, te)


def score_method(method: str, inputs, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    if method.startswith(("probe_", "embed_")):
        return score_dense_lr(inputs, y, tr, te)[0]
    if method.startswith("diffmean_"):
        return score_diffmean(inputs, y, tr, te)
    if method.startswith("tfidf_"):
        return score_tfidf(inputs, y, tr, te)
    if method == "protectai_document":  # zero-shot: no training
        return inputs[te]
    raise ValueError(method)


class OutOfTime(Exception):
    """Raised when the --budget-min wall-clock budget is used up (rerun to resume)."""


class JobRunner:
    """Runs jobs in parallel batches and caches each result on disk.

    Each job is ``(key, func, make_args)``; ``make_args`` is called lazily so large
    inputs are only built for the batch being run. Finished jobs are skipped on the
    next invocation, so an interrupted evaluation resumes where it stopped.
    """

    def __init__(self, cache_dir: Path, n_jobs: int, deadline: float | None):
        self.cache_dir, self.n_jobs, self.deadline = cache_dir, n_jobs, deadline
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def run(self, jobs: list[tuple[str, object, object]]) -> dict[str, object]:
        todo = [j for j in jobs if not self._path(j[0]).exists()]
        step = max(1, 2 * self.n_jobs)
        with Parallel(n_jobs=self.n_jobs) as par:
            for i in range(0, len(todo), step):
                if self.deadline is not None and time.time() > self.deadline:
                    raise OutOfTime()
                batch = todo[i : i + step]
                results = par(delayed(func)(*make_args()) for _, func, make_args in batch)
                for (key, _, _), r in zip(batch, results):
                    with open(self._path(key), "wb") as fh:
                        pickle.dump(r, fh)
        out = {}
        for key, _, _ in jobs:
            with open(self._path(key), "rb") as fh:
                out[key] = pickle.load(fh)
        return out


def method_key(method: str, layers: dict[str, int]) -> str:
    if method.startswith(("probe_", "diffmean_")):
        return f"{method}_L{layers[method.split('_')[1]]}"
    return method


def score_jobs(exp: Experiment, prefix: str, splits: list[tuple[str, np.ndarray, np.ndarray]],
               methods: list[str], layers: dict[str, int], y: np.ndarray | None = None):
    """Jobs scoring every method on every (split_name, train_idx, test_idx)."""
    y = exp.y if y is None else y
    inputs: dict[str, object] = {}

    def args_for(m, tr, te):
        def make():
            if m not in inputs:
                inputs[m] = exp.inputs(m, layers)
            return (m, inputs[m], y, tr, te)
        return make

    return [(f"{prefix}_{name}_{method_key(m, layers)}", score_method, args_for(m, tr, te))
            for name, tr, te in splits for m in methods]


def _sweep_one(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds, kind: str) -> dict:
    per_fold = []
    for tr, te in folds:
        if kind == "lr":
            s, C = score_dense_lr(X, y, tr, te)
        else:
            s, C = score_diffmean(X, y, tr, te), float("nan")
        m = metrics(y[te], s, groups[te])
        m["C"] = C
        per_fold.append(m)
    return pd.DataFrame(per_fold).mean(numeric_only=True).to_dict()


def layer_sweep(exp: Experiment, folds, runner: JobRunner) -> pd.DataFrame:
    specs = [(pos, layer, kind) for pos in ("prompt", "response") for kind in ("lr", "diffmean")
             for layer in range(exp.n_layers)]

    def make(pos, layer, kind):
        return lambda: (np.ascontiguousarray(exp.acts[pos][:, layer, :]), exp.y, exp.groups, folds, kind)

    jobs = [(f"sweep_{pos}_{layer:02d}_{kind}", _sweep_one, make(pos, layer, kind)) for pos, layer, kind in specs]
    res = runner.run(jobs)
    rows = [{"position": pos, "layer": layer, "probe": kind, **res[key]}
            for (pos, layer, kind), (key, _, _) in zip(specs, jobs)]
    return pd.DataFrame(rows)


def holdout_docs(df: pd.DataFrame, per_genre: int = 2) -> set[str]:
    rng = np.random.default_rng(SEED)
    docs = set()
    for _, sub in df.groupby("genre"):
        ids = sorted(sub["doc_id"].unique())
        docs.update(rng.choice(ids, size=per_genre, replace=False).tolist())
    return docs


def generalization_splits(df: pd.DataFrame):
    """Yield (split_type, held_out_value, train_idx, test_idx)."""
    hold = df["doc_id"].isin(holdout_docs(df)).to_numpy()
    cond = df["condition"].to_numpy()
    for p in PAYLOADS:
        is_p = (df["payload"] == p).to_numpy()
        te = is_p | (np.isin(cond, ["clean", "benign_instruction"]) & hold)
        yield "payload", p, np.where(~te)[0], np.where(te)[0]
    for f in FRAMINGS:
        is_f = (df["framing"] == f).to_numpy()
        te = is_f | ((cond != "injected") & hold)
        yield "framing", f, np.where(~te)[0], np.where(te)[0]


def breakdown(df: pd.DataFrame, oof: dict[str, np.ndarray], by: str) -> pd.DataFrame:
    """Pooled out-of-fold AUROC of positives with attribute value v against matched negatives."""
    rows = []
    y, g = df["y_obey"].to_numpy(), df["group"].to_numpy()
    for v in sorted(df.loc[df["condition"] == "injected", by].dropna().unique()):
        pos = (y == 1) & (df[by] == v).to_numpy()
        ign = (g == "injected_ignored") & (df[by] == v).to_numpy()
        if by == "payload":
            usr = (g == "user_followed") & (df["payload"] == v).to_numpy()
        else:
            usr = g == "user_followed"
        for method, s in oof.items():
            r = {by: v, "method": method, "n_pos": int(pos.sum()), "n_ignored": int(ign.sum()), "n_user_followed": int(usr.sum())}
            for name, neg in (("vs_injected_ignored", ign), ("vs_user_followed", usr)):
                m = pos | neg
                r[f"auroc_{name}"] = _auc(y[m], s[m]) if pos.any() and neg.any() else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def behavior_tables(df_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Follow rates (ambiguous verdicts included in the denominators, reported separately)."""
    out = {}
    inj = df_all[df_all["condition"] == "injected"]
    for col in ("payload", "framing", "placement", "system", "task"):
        t = inj.groupby(col)["behavior"].value_counts().unstack(fill_value=0)
        t["n"] = t.sum(axis=1)
        t["follow_rate"] = t.get("followed", 0) / t["n"]
        out[f"injected_by_{col}"] = t.reset_index()
    t = inj.pivot_table(index="payload", columns="framing", values="behavior",
                        aggfunc=lambda b: float((b == "followed").mean()))
    out["injected_payload_x_framing"] = t.reset_index()
    for cond in ("user_payload", "payload_mention"):
        sub = df_all[df_all["condition"] == cond]
        t = sub.groupby("payload")["behavior"].value_counts().unstack(fill_value=0)
        t["n"] = t.sum(axis=1)
        t["follow_rate"] = t.get("followed", 0) / t["n"]
        out[f"{cond}_by_payload"] = t.reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-generalization", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="per-job result cache (default: <run-dir>/eval_cache)")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="stop cleanly (exit code 3) after this many minutes; rerun to resume from the cache")
    args = ap.parse_args()
    try:
        evaluate(args)
    except OutOfTime:
        print("INCOMPLETE: time budget used up; rerun the same command to resume", flush=True)
        raise SystemExit(3)


def evaluate(args) -> None:
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.run_dir) / "eval_cache"
    deadline = t0 + 60 * args.budget_min if args.budget_min else None
    runner = JobRunner(cache_dir, args.n_jobs, deadline)

    df_all = load_labeled(args.run_dir, args.prompts, keep_ambiguous=True)
    df = df_all[df_all["group"].notna()].reset_index(drop=True)
    acts = load_activations(args.run_dir, df["id"].tolist())
    ok = ~np.isnan(acts["response"]).any(axis=(1, 2))  # drop empty responses
    if not ok.all():
        df = df[ok].reset_index(drop=True)
        acts = {k: v[ok] for k, v in acts.items()}
    tf_npz = np.load(Path(args.run_dir) / "text_features.npz", allow_pickle=True)
    tf_pos = {i: k for k, i in enumerate(tf_npz["ids"].tolist())}
    rows = np.array([tf_pos[i] for i in df["id"]])
    text_feats = {k: tf_npz[k][rows] for k in ("emb_prompt", "emb_response", "protectai")}
    exp = Experiment(df, acts, text_feats)
    run_meta = json.loads((Path(args.run_dir) / "run_meta.json").read_text())

    # ---- dataset summary
    keep = ["id", "condition", "group", "behavior", "payload", "framing", "placement", "system", "task",
            "doc_id", "inserted", "response", "n_response_tokens", "finished"]
    df_all[keep].to_json(out / "labeled_examples.jsonl", orient="records", lines=True, force_ascii=False)
    for name, t in behavior_tables(df_all).items():
        t.to_csv(out / f"behavior_{name}.csv", index=False)
    counts = df["group"].value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict()
    n_ambiguous = int((df_all["group"].isna()).sum())
    print(f"examples: {len(df)} (dropped {n_ambiguous} ambiguous, {int((~ok).sum())} empty); groups: {counts}", flush=True)

    # ---- 1. layer sweep
    folds = exp.cv_folds()
    sweep = layer_sweep(exp, folds, runner)
    sweep.to_csv(out / "layer_sweep.csv", index=False)
    lr = sweep[sweep["probe"] == "lr"]
    layers = {pos: int(lr[lr["position"] == pos].sort_values("auroc", ascending=False)["layer"].iloc[0])
              for pos in ("prompt", "response")}
    print(f"[{(time.time() - t0) / 60:.1f} min] layer sweep done; best layers {layers}", flush=True)

    # ---- 2. main comparison (same folds)
    methods = list(METHOD_LABELS)
    oof = {m: np.zeros(len(df)) for m in methods}
    fold_rows = []
    fold_of = np.zeros(len(df), dtype=int)
    fold_splits = [(f"fold{k}", tr, te) for k, (tr, te) in enumerate(folds)]
    scores = runner.run(score_jobs(exp, "main", fold_splits, methods, layers))
    for k, (tr, te) in enumerate(folds):
        fold_of[te] = k
        for m in methods:
            s = scores[f"main_fold{k}_{method_key(m, layers)}"]
            oof[m][te] = s
            fold_rows.append({"method": m, "fold": k, **metrics(exp.y[te], s, exp.groups[te])})
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out / "main_per_fold.csv", index=False)
    main_tbl = folds_df.drop(columns="fold").groupby("method", sort=False).agg(["mean", "std"])
    main_tbl.columns = [f"{a}_{b}" for a, b in main_tbl.columns]
    main_tbl = main_tbl.reset_index()
    main_tbl.insert(1, "label", main_tbl["method"].map(METHOD_LABELS))
    main_tbl.to_csv(out / "main_comparison.csv", index=False)
    oof_df = df[["id", "group", "condition", "payload", "framing", "placement", "system", "task", "doc_id"]].copy()
    oof_df["fold"] = fold_of
    for m in methods:
        oof_df[m] = oof[m]
    oof_df.to_csv(out / "oof_scores.csv", index=False)
    breakdown(df, oof, "framing").to_csv(out / "breakdown_by_framing.csv", index=False)
    breakdown(df, oof, "payload").to_csv(out / "breakdown_by_payload.csv", index=False)

    # Same folds, secondary target: is an injection present at all (followed or not)?
    y_inj = df["y_injected"].to_numpy()
    presence_methods = ["probe_prompt", "tfidf_prompt", "embed_prompt", "protectai_document"]
    scores = runner.run(score_jobs(exp, "presence", fold_splits, presence_methods, layers, y=y_inj))
    inj_rows = [{"method": m, "auroc": _auc(y_inj[te], scores[f"presence_fold{k}_{method_key(m, layers)}"])}
                for k, (tr, te) in enumerate(folds) for m in presence_methods]
    inj_tbl = pd.DataFrame(inj_rows).groupby("method", sort=False)["auroc"].agg(["mean", "std"]).reset_index()
    inj_tbl.to_csv(out / "injection_presence.csv", index=False)
    print(f"[{(time.time() - t0) / 60:.1f} min] main comparison done", flush=True)

    # ---- 3. generalization
    if not args.skip_generalization:
        splits = list(generalization_splits(df))
        named = [(f"{split_type}-{held}", tr, te) for split_type, held, tr, te in splits]
        scores = runner.run(score_jobs(exp, "gen", named, methods, layers))
        gen_rows = []
        for (split_type, held, tr, te), (name, _, _) in zip(splits, named):
            for m in methods:
                s = scores[f"gen_{name}_{method_key(m, layers)}"]
                gen_rows.append({"split": split_type, "held_out": held, "method": m,
                                 **metrics(exp.y[te], s, exp.groups[te])})
        pd.DataFrame(gen_rows).to_csv(out / "generalization.csv", index=False)
        print(f"[{(time.time() - t0) / 60:.1f} min] generalization done", flush=True)

    summary = {
        "model": run_meta["model"],
        "n_examples": int(len(df)),
        "n_dropped_ambiguous": n_ambiguous,
        "group_counts": counts,
        "injected_follow_rate": float((df_all.loc[df_all["condition"] == "injected", "behavior"] == "followed").mean()),
        "user_payload_follow_rate": float((df_all.loc[df_all["condition"] == "user_payload", "behavior"] == "followed").mean()),
        "best_layers": layers,
        "n_layers_plus_embeddings": exp.n_layers,
        "folds": N_FOLDS,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    cols = ["label", "auroc_mean", "auroc_std", "tpr_at_5fpr_mean"] + [f"auroc_vs_{g}_mean" for g in NEGATIVE_GROUPS]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(main_tbl[cols].to_string(index=False))


if __name__ == "__main__":
    main()
