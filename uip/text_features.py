"""Precompute text-only features for the baselines.

* ``emb_prompt`` / ``emb_response``: sentence embeddings from BAAI/bge-small-en-v1.5
  of the full prompt (system prompt + user message) and of the model's response.
* ``protectai``: P(injection) from the off-the-shelf classifier
  protectai/deberta-v3-base-prompt-injection-v2, applied zero-shot to the document
  text (the untrusted input), as such classifiers are usually deployed.

TF-IDF features are cheap and are built inside ``uip.evaluate`` per fold.

Usage::

    python -m uip.text_features --run-dir runs/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uip.labels import load_labeled

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INJECTION_CLASSIFIER = "protectai/deberta-v3-base-prompt-injection-v2"


@torch.inference_mode()
def injection_scores(texts: list[str], batch_size: int = 16) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(INJECTION_CLASSIFIER)
    clf = AutoModelForSequenceClassification.from_pretrained(INJECTION_CLASSIFIER).eval()
    inj = [i for i, name in clf.config.id2label.items() if "INJECTION" in name.upper()]
    assert len(inj) == 1, clf.config.id2label
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], truncation=True, max_length=512, padding=True, return_tensors="pt")
        out.append(torch.softmax(clf(**enc).logits.float(), dim=-1)[:, inj[0]].numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    df = load_labeled(args.run_dir, args.prompts, keep_ambiguous=True)
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    kw = dict(batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_prompt = embedder.encode(df["prompt_text"].tolist(), **kw)
    emb_response = embedder.encode(df["response"].tolist(), **kw)
    protectai = injection_scores(df["document"].tolist())

    out = Path(args.run_dir) / "text_features.npz"
    np.savez(
        out,
        ids=df["id"].to_numpy(),
        emb_prompt=emb_prompt.astype(np.float32),
        emb_response=emb_response.astype(np.float32),
        protectai=protectai.astype(np.float32),
    )
    print(f"saved {out}: {len(df)} examples, embedding dim {emb_prompt.shape[1]}")


if __name__ == "__main__":
    main()
