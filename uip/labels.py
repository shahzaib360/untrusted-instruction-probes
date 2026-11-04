"""Join prompts, generations and checker verdicts into one labelled table.

Target for the main task (``y_obey``):

    1  the payload was injected into the document AND the response carried it out
    0  everything else: clean documents, benign human-directed instructions,
       payload mentions, the user asking for the payload, injections the model ignored

Injected examples whose checker verdict is ``ambiguous`` are dropped.

``group`` gives a finer breakdown used for per-negative-type metrics:
``injected_followed`` (the positives), ``injected_ignored``, ``user_followed``,
``user_not_followed``, ``payload_mention``, ``benign_instruction``, ``clean``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from uip.checkers import AMBIGUOUS, FOLLOWED, check

GROUPS = [
    "injected_followed",
    "injected_ignored",
    "user_followed",
    "user_not_followed",
    "payload_mention",
    "benign_instruction",
    "clean",
]
NEGATIVE_GROUPS = GROUPS[1:]

_DOC_RE = re.compile(r"<document>\n(.*)\n</document>\s*$", re.DOTALL)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def document_text(user_message: str) -> str:
    """The untrusted part of the prompt (everything inside the document tags)."""
    m = _DOC_RE.search(user_message)
    return m.group(1) if m else user_message


def _group(condition: str, behavior: str) -> str | None:
    if condition == "injected":
        if behavior == AMBIGUOUS:
            return None
        return "injected_followed" if behavior == FOLLOWED else "injected_ignored"
    if condition == "user_payload":
        return "user_followed" if behavior == FOLLOWED else "user_not_followed"
    return condition


def load_labeled(run_dir: str | Path, prompts_path: str | Path = "data/prompts.jsonl", keep_ambiguous: bool = False) -> pd.DataFrame:
    run_dir = Path(run_dir)
    df = pd.DataFrame(load_jsonl(prompts_path))
    gens = {g["id"]: g for g in load_jsonl(run_dir / "generations.jsonl")}
    df = df[df["id"].isin(gens)].copy()
    for key in ("response", "n_prompt_tokens", "n_response_tokens", "finished"):
        df[key] = df["id"].map(lambda i: gens[i][key])
    df["behavior"] = [
        check(p, r) if isinstance(p, str) else "n/a" for p, r in zip(df["payload"], df["response"])
    ]
    df["group"] = [_group(c, b) for c, b in zip(df["condition"], df["behavior"])]
    if not keep_ambiguous:
        df = df[df["group"].notna()].copy()
    df["y_obey"] = (df["group"] == "injected_followed").astype(int)
    df["y_injected"] = (df["condition"] == "injected").astype(int)
    df["prompt_text"] = df["system_prompt"] + "\n\n" + df["user_message"]
    df["document"] = df["user_message"].map(document_text)
    return df.reset_index(drop=True)


def load_activations(run_dir: str | Path, ids: list[str]) -> dict[str, np.ndarray]:
    """Activations for ``ids`` (in that order): {'prompt': [N,L+1,d], 'response': [N,L+1,d]}."""
    run_dir = Path(run_dir)
    all_ids = json.loads((run_dir / "acts_ids.json").read_text())
    pos = {i: k for k, i in enumerate(all_ids)}
    rows = np.array([pos[i] for i in ids])
    out = {}
    for name, fname in (("prompt", "acts_prompt_last.npy"), ("response", "acts_response_mean.npy")):
        arr = np.load(run_dir / fname, mmap_mode="r")
        out[name] = np.asarray(arr[rows], dtype=np.float32)
    return out
