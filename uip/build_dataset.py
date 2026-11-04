"""Assemble the prompt set used in all experiments.

Five conditions (``condition`` column):

* ``clean``              - task + ordinary document.
* ``benign_instruction`` - the document contains an instruction aimed at a human
                           reader ("Please submit your expense report by Friday").
* ``payload_mention``    - the document *mentions* a payload's content without
                           instructing anyone ("...printed entirely in capital letters").
* ``user_payload``       - the *user* asks for the payload behaviour (trusted).
* ``injected``           - the payload instruction is planted inside the document.

Only ``injected`` examples whose response actually carries out the payload are
positives for the main detection task; see ``uip.labels``.

Usage::

    python -m uip.build_dataset --out data/prompts.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from uip.corpus import (
    BENIGN_INSTRUCTIONS,
    FRAMINGS,
    GENRE_NOUN,
    PAYLOADS,
    PLACEMENTS,
    SYSTEM_PROMPTS,
    TASKS,
    all_documents,
    insert_sentence,
)


def user_message(task_text: str, doc_text: str, extra_instruction: str | None = None) -> str:
    head = task_text if extra_instruction is None else f"{task_text} {extra_instruction}"
    return f"{head}\n\n<document>\n{doc_text}\n</document>"


def build(
    seed: int = 0,
    n_injected_per_cell: int = 10,
    n_user_per_payload: int = 60,
    n_mention_per_payload: int = 40,
    n_benign: int = 300,
    n_clean: int = 300,
) -> list[dict]:
    rng = random.Random(seed)
    docs = all_documents()
    task_names = list(TASKS)
    sys_names = list(SYSTEM_PROMPTS)
    rows: list[dict] = []

    def base_row(condition: str, doc: dict, task: str, system: str) -> dict:
        return {
            "condition": condition,
            "doc_id": doc["doc_id"],
            "genre": doc["genre"],
            "task": task,
            "system": system,
            "payload": None,
            "framing": None,
            "placement": None,
            "phrasing": None,
            "inserted": None,
            "system_prompt": SYSTEM_PROMPTS[system],
        }

    def task_text(task: str, doc: dict) -> str:
        return TASKS[task].format(noun=GENRE_NOUN[doc["genre"]])

    # clean: distinct (doc, task, system) combinations
    combos = list(itertools.product(range(len(docs)), task_names, sys_names))
    rng.shuffle(combos)
    for di, task, system in combos[:n_clean]:
        doc = docs[di]
        row = base_row("clean", doc, task, system)
        row["user_message"] = user_message(task_text(task, doc), doc["text"])
        rows.append(row)

    # benign_instruction: a human-directed instruction inside the document
    for _ in range(n_benign):
        doc, task, system = rng.choice(docs), rng.choice(task_names), rng.choice(sys_names)
        sentence = rng.choice(BENIGN_INSTRUCTIONS)
        placement = rng.choice(PLACEMENTS)
        row = base_row("benign_instruction", doc, task, system)
        row.update(placement=placement, inserted=sentence)
        row["user_message"] = user_message(task_text(task, doc), insert_sentence(doc["text"], sentence, placement))
        rows.append(row)

    for payload, spec in PAYLOADS.items():
        # payload_mention: payload content appears, but not as an instruction
        for _ in range(n_mention_per_payload):
            doc, task, system = rng.choice(docs), rng.choice(task_names), rng.choice(sys_names)
            k = rng.randrange(len(spec["mentions"]))
            sentence = spec["mentions"][k]
            placement = rng.choice(PLACEMENTS)
            row = base_row("payload_mention", doc, task, system)
            row.update(payload=payload, placement=placement, phrasing=k, inserted=sentence)
            row["user_message"] = user_message(
                task_text(task, doc), insert_sentence(doc["text"], sentence, placement)
            )
            rows.append(row)

        # user_payload: the trusted user asks for the payload behaviour
        for _ in range(n_user_per_payload):
            doc, task, system = rng.choice(docs), rng.choice(task_names), rng.choice(sys_names)
            k = rng.randrange(len(spec["instructions"]))
            instr = spec["instructions"][k]
            row = base_row("user_payload", doc, task, system)
            row.update(payload=payload, phrasing=k, inserted=instr)
            row["user_message"] = user_message(task_text(task, doc), doc["text"], extra_instruction=instr)
            rows.append(row)

        # injected: payload planted in the document, balanced over framing x placement
        for framing, placement in itertools.product(FRAMINGS, PLACEMENTS):
            for _ in range(n_injected_per_cell):
                doc, task, system = rng.choice(docs), rng.choice(task_names), rng.choice(sys_names)
                k = rng.randrange(len(spec["instructions"]))
                injection = FRAMINGS[framing].format(instr=spec["instructions"][k])
                row = base_row("injected", doc, task, system)
                row.update(payload=payload, framing=framing, placement=placement, phrasing=k, inserted=injection)
                row["user_message"] = user_message(
                    task_text(task, doc), insert_sentence(doc["text"], injection, placement)
                )
                rows.append(row)

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"ex{i:05d}"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/prompts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-injected-per-cell", type=int, default=10)
    ap.add_argument("--n-user-per-payload", type=int, default=60)
    ap.add_argument("--n-mention-per-payload", type=int, default=40)
    ap.add_argument("--n-benign", type=int, default=300)
    ap.add_argument("--n-clean", type=int, default=300)
    args = ap.parse_args()

    rows = build(
        seed=args.seed,
        n_injected_per_cell=args.n_injected_per_cell,
        n_user_per_payload=args.n_user_per_payload,
        n_mention_per_payload=args.n_mention_per_payload,
        n_benign=args.n_benign,
        n_clean=args.n_clean,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["condition"]] = counts.get(row["condition"], 0) + 1
    print(f"wrote {len(rows)} prompts to {out}: {counts}")


if __name__ == "__main__":
    main()
