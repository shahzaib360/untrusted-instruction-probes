"""Run the model on every prompt: greedy-decode a response and record activations.

For every example and every layer (HF ``hidden_states``: index 0 is the
embedding output, index L the final post-norm residual stream) we save:

* ``prompt_last``   - the residual stream at the last prompt token, i.e. right
  before the model writes its first response token. A probe here has to
  *predict* whether the model is about to obey the injected text.
* ``response_mean`` - the residual stream averaged over the response tokens the
  model generates. A probe here *detects* obedience while the model answers.

Both are collected in a single pass: the prompt states come from the prefill
step of ``generate`` and the response states from the decoding steps.

Outputs in ``--out-dir``::

    generations.jsonl            id, response, token counts, finished flag
    chunks/chunk_XXXXX.npz       per-chunk activations (resumable)
    acts_prompt_last.npy         [N, L+1, d] float32, rows ordered as acts_ids.json
    acts_response_mean.npy       [N, L+1, d] float32
    acts_ids.json

Usage::

    python -m uip.generate --model Qwen/Qwen2.5-1.5B-Instruct --out-dir runs/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_text(tok, row: dict) -> str:
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_message"]},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def run_batch(model, tok, texts: list[str], gen_cfg: GenerationConfig, eos_ids: list[int]):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    out = model.generate(**enc, generation_config=gen_cfg)
    T = enc["input_ids"].shape[1]
    seqs = out.sequences[:, T:]  # [B, S] generated tokens
    hs = out.hidden_states  # tuple over decoding steps -> tuple over layers
    n_layers = len(hs[0])
    B, S = seqs.shape

    # Prefill step: last position is the true last prompt token (left padding).
    prompt_last = torch.stack([hs[0][l][:, -1, :] for l in range(n_layers)], dim=1).float()

    # Number of content tokens before the first EOS (S if the answer was cut off).
    is_eos = torch.isin(seqs, torch.tensor(eos_ids))
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(has_eos, is_eos.int().argmax(dim=1), torch.full((B,), S))
    # Decoding step t (t >= 1) processes the t-th generated token, so content
    # tokens 1..n are covered by steps 1..min(n, S-1).
    n_steps = torch.clamp(first_eos, max=S - 1)

    d = prompt_last.shape[-1]
    resp_sum = torch.zeros(B, n_layers, d)
    for t in range(1, S):
        valid = (n_steps >= t).float().view(B, 1, 1)
        if valid.sum() == 0:
            break
        step = torch.stack([hs[t][l][:, -1, :] for l in range(n_layers)], dim=1).float()
        resp_sum += valid * step
    denom = n_steps.clamp(min=1).float().view(B, 1, 1)
    response_mean = resp_sum / denom
    response_mean[n_steps == 0] = float("nan")

    responses = [tok.decode(seqs[b, : first_eos[b]], skip_special_tokens=True) for b in range(B)]
    meta = [
        {
            "n_prompt_tokens": int(enc["attention_mask"][b].sum()),
            "n_response_tokens": int(first_eos[b]),
            "finished": bool(has_eos[b]),
        }
        for b in range(B)
    ]
    return responses, meta, prompt_last.numpy(), response_mean.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N prompts (pilot)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    eos_ids = model.generation_config.eos_token_id
    eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Pure greedy decoding. Every field is set explicitly because transformers
    # fills unset fields from the model's own generation_config, which for Qwen2.5
    # includes repetition_penalty=1.1. That penalty also applies to prompt tokens,
    # so it would discourage copying injected strings (canary words, URLs) and
    # bias exactly the behaviour we measure.
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    texts = {row["id"]: chat_text(tok, row) for row in rows}
    lengths = {i: len(tok(t, add_special_tokens=False)["input_ids"]) for i, t in texts.items()}
    # Group by task (similar answer lengths, so batches stop decoding together),
    # then by prompt length (less padding).
    task_of = {row["id"]: row["task"] for row in rows}
    order = sorted(texts, key=lambda i: (task_of[i], lengths[i], i))
    chunks = [order[i : i + args.chunk_size] for i in range(0, len(order), args.chunk_size)]

    print(f"{len(rows)} prompts, {len(chunks)} chunks, model={args.model}", flush=True)
    t_start = time.time()
    done_examples = 0
    for ci, chunk_ids in enumerate(chunks):
        path = chunk_dir / f"chunk_{ci:05d}.npz"
        if path.exists():
            done_examples += len(chunk_ids)
            continue
        t0 = time.time()
        gens, pl, rm = [], [], []
        for bi in range(0, len(chunk_ids), args.batch_size):
            ids = chunk_ids[bi : bi + args.batch_size]
            responses, meta, p_last, r_mean = run_batch(model, tok, [texts[i] for i in ids], gen_cfg, eos_ids)
            for i, r, m in zip(ids, responses, meta):
                gens.append({"id": i, "response": r, **m})
            pl.append(p_last)
            rm.append(r_mean)
        np.savez(
            path,
            ids=np.array(chunk_ids),
            prompt_last=np.concatenate(pl).astype(np.float32),
            response_mean=np.concatenate(rm).astype(np.float32),
            generations=np.array(json.dumps(gens, ensure_ascii=False)),
        )
        done_examples += len(chunk_ids)
        elapsed = time.time() - t_start
        rate = len(chunk_ids) / (time.time() - t0)
        remaining = (len(order) - done_examples) / max(rate, 1e-9)
        print(
            f"chunk {ci + 1}/{len(chunks)}: {rate:.2f} ex/s, prompt len ~{lengths[chunk_ids[-1]]}, "
            f"elapsed {elapsed / 60:.1f} min, eta {remaining / 60:.1f} min",
            flush=True,
        )

    # Merge chunks into single arrays.
    ids_all, pl_all, rm_all, gens_all = [], [], [], []
    for ci in range(len(chunks)):
        z = np.load(chunk_dir / f"chunk_{ci:05d}.npz")
        ids_all.extend(z["ids"].tolist())
        pl_all.append(z["prompt_last"])
        rm_all.append(z["response_mean"])
        gens_all.extend(json.loads(str(z["generations"])))
    np.save(out_dir / "acts_prompt_last.npy", np.concatenate(pl_all))
    np.save(out_dir / "acts_response_mean.npy", np.concatenate(rm_all))
    (out_dir / "acts_ids.json").write_text(json.dumps(ids_all))
    by_id = {g["id"]: g for g in gens_all}
    with (out_dir / "generations.jsonl").open("w") as f:
        for row in rows:  # keep prompt-file order
            f.write(json.dumps(by_id[row["id"]], ensure_ascii=False) + "\n")
    meta = {
        "model": args.model,
        "n_examples": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "n_layers_plus_embeddings": int(pl_all[0].shape[1]),
        "d_model": int(pl_all[0].shape[2]),
        "total_minutes": round((time.time() - t_start) / 60, 1),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
