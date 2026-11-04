from collections import Counter

from uip.build_dataset import build
from uip.corpus import FRAMINGS, PAYLOADS, PLACEMENTS, insert_sentence, split_sentences
from uip.labels import document_text


def test_counts_and_balance():
    rows = build(seed=0)
    counts = Counter(r["condition"] for r in rows)
    assert counts == {
        "injected": len(PAYLOADS) * len(FRAMINGS) * len(PLACEMENTS) * 10,
        "user_payload": len(PAYLOADS) * 60,
        "payload_mention": len(PAYLOADS) * 40,
        "benign_instruction": 300,
        "clean": 300,
    }
    cells = Counter((r["payload"], r["framing"], r["placement"]) for r in rows if r["condition"] == "injected")
    assert set(cells.values()) == {10}
    assert len({r["id"] for r in rows}) == len(rows)


def test_deterministic():
    assert build(seed=0) == build(seed=0)
    assert build(seed=0) != build(seed=1)


def test_where_the_instruction_lives():
    for r in build(seed=0):
        doc = document_text(r["user_message"])
        if r["condition"] == "injected":
            # the payload instruction sits inside the untrusted document
            instr = r["inserted"].split("\n\n")[1] if r["framing"] == "fake_end" else r["inserted"]
            assert instr in doc
        elif r["condition"] == "user_payload":
            # ...or in the user's own request, outside the document
            head = r["user_message"].split("<document>")[0]
            assert r["inserted"] in head and r["inserted"] not in doc
        elif r["condition"] in ("benign_instruction", "payload_mention"):
            assert r["inserted"] in doc
        else:
            assert r["inserted"] is None


def test_insert_sentence_placements():
    doc = "One. Two. Three. Four."
    assert insert_sentence(doc, "X.", "start") == "X. One. Two. Three. Four."
    assert insert_sentence(doc, "X.", "middle") == "One. Two. X. Three. Four."
    assert insert_sentence(doc, "X.", "end") == "One. Two. Three. Four. X."
    assert split_sentences(doc) == ["One.", "Two.", "Three.", "Four."]
