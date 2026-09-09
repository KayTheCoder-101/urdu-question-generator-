# A.2 + A.3 — Load UQA dataset and prepare training data

from datasets import load_dataset
import csv
import matplotlib.pyplot as plt

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
SENT_DELIMS = "\u06D4\u061F!"

# Split context into sentences
def split_sentences(text):
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start:i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]

# Create one source-target pair
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

            src = (
                sent[:rel]
                + " " + ANS_OPEN + " " + a_text + " "
                + ANS_CLOSE + " "
                + sent[rel + len(a_text):]
            ).strip()

            src = " ".join(src.split())
            tgt = " ".join(example["question"].split())

            if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
                return None

            return src, tgt

    return None

# Build and save a dataset split — NOW also returns lengths
def build_split(split, out_path):
    pairs = [p for p in map(make_pair, split) if p is not None]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)

    print(f"{out_path}: {len(pairs)} pairs")

    # NEW: compute lengths from the pairs we just built
    src_lengths = []
    tgt_lengths = []
    for src, tgt in pairs:
        src_lengths.append(len(src.split()))
        tgt_lengths.append(len(tgt.split()))

    return pairs, src_lengths, tgt_lengths

# Plot and save length histograms
def plot_length_histograms(src_lengths, tgt_lengths, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(src_lengths, bins=30)
    axes[0].set_title("Source length distribution")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")

    axes[1].hist(tgt_lengths, bins=30, color="orange")
    axes[1].set_title("Target length distribution")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    ds = load_dataset("uqa/UQA")
    print(ds)

    ex = ds["train"][0]
    print(ex.keys())
    print(ex["question"])
    print(ex["answer"])

    n_total = len(ds["train"])
    n_ans = sum(not ex["is_impossible"] for ex in ds["train"])
    print(f"train rows: {n_total}, answerable: {n_ans}")

    train_pairs, train_src_lens, train_tgt_lens = build_split(ds["train"], "data/train.tsv")
    valid_pairs, valid_src_lens, valid_tgt_lens = build_split(ds["validation"], "data/valid.tsv")

    plot_length_histograms(train_src_lens, train_tgt_lens, "results/length_histograms.png")