# Person A/B — Task 1: Data preparation (tested, working version)
from datasets import load_dataset
import csv

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
SENT_DELIMS = "\u06D4\u061F!"

def split_sentences(text):
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start:i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]

def make_pair(example, max_src=60, max_tgt=25):
    a_text = example["answer"]
    if not a_text:
        return None
    context = example["context"]
    a_start = context.find(a_text)
    if a_start == -1:
        return None
    for s, e, sent in split_sentences(context):
        if s <= a_start < e:
            rel = a_start - s
            if sent[rel:rel + len(a_text)] != a_text:
                return None
            src = (sent[:rel] + " " + ANS_OPEN + " " + a_text + " "
                   + ANS_CLOSE + " " + sent[rel + len(a_text):]).strip()
            src = " ".join(src.split())
            tgt = " ".join(example["question"].split())
            if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
                return None
            return src, tgt
    return None

def build_split(split, out_path):
    pairs = [p for p in map(make_pair, split) if p is not None]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)
    print(f"{out_path}: {len(pairs)} pairs")
    return pairs

if __name__ == "__main__":
    ds = load_dataset("uqa/UQA")
    train_pairs = build_split(ds["train"], "data/train.tsv")
    valid_pairs = build_split(ds["validation"], "data/valid.tsv")