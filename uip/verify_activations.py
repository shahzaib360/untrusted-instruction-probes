"""Sanity check: activations captured during generation match a teacher-forced pass.

``uip.generate`` collects activations from inside ``model.generate`` (batched,
left-padded, KV-cached). This script recomputes them for a few examples with a
plain, unbatched forward pass over ``prompt + response`` and reports the cosine
similarity per layer. Values close to 1 mean the bookkeeping (which position is
the last prompt token, which steps belong to the response) is right.

Usage::

    python -m uip.verify_activations --run-dir runs/qwen2.5-1.5b --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uip.generate import chat_text, load_jsonl


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    run = Path(args.run_dir)
    ids = json.loads((run / "acts_ids.json").read_text())
    pl = np.load(run / "acts_prompt_last.npy", mmap_mode="r")
    rm = np.load(run / "acts_response_mean.npy", mmap_mode="r")
    gens = {g["id"]: g for g in load_jsonl(run / "generations.jsonl")}
    prompts = {r["id"]: r for r in load_jsonl(args.prompts)}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()

    rng = np.random.default_rng(0)
    worst_pl, worst_rm = 1.0, 1.0
    for idx in rng.choice(len(ids), size=min(args.n, len(ids)), replace=False):
        ex_id = ids[idx]
        g = gens[ex_id]
        prompt_ids = tok(chat_text(tok, prompts[ex_id]), add_special_tokens=False)["input_ids"]
        resp_ids = tok(g["response"], add_special_tokens=False)["input_ids"]
        full = torch.tensor([prompt_ids + resp_ids])
        hs = model(full, output_hidden_states=True).hidden_states
        P = len(prompt_ids)
        tf_pl = torch.stack([h[0, P - 1] for h in hs]).float().numpy()
        # generation covers response tokens 1..n (minus the last one if cut off)
        n = g["n_response_tokens"] if g["finished"] else g["n_response_tokens"] - 1
        tf_rm = torch.stack([h[0, P : P + n].mean(0) for h in hs]).float().numpy()
        c_pl = cosine(tf_pl[1:], np.asarray(pl[idx])[1:])
        c_rm = cosine(tf_rm[1:], np.asarray(rm[idx])[1:])
        worst_pl, worst_rm = min(worst_pl, c_pl.min()), min(worst_rm, c_rm.min())
        retok = len(resp_ids) == g["n_response_tokens"]
        print(
            f"{ex_id}: prompt_last cos min {c_pl.min():.4f} mean {c_pl.mean():.4f} | "
            f"response_mean cos min {c_rm.min():.4f} mean {c_rm.mean():.4f} | retokenization exact: {retok}"
        )
    print(f"worst layer cosine: prompt_last {worst_pl:.4f}, response_mean {worst_rm:.4f}")


if __name__ == "__main__":
    main()
