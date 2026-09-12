import csv
import sacrebleu
import torch
import torch.nn as nn
from rouge_score import rouge_scorer

from decode import greedy_decode, beam_search, enc, dec, sp, device


# Whitespace-based tokenizer for Urdu (rouge_score's default tokenizer
# discards non-Latin characters, which breaks scoring on Urdu text)
class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


# Load test examples
def load_pairs(tsv_path, n=50):
    pairs = []

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))

    return pairs[:n]


# Calculate unknown token rate
def unk_rate(hyps):
    total = sum(len(h.split()) for h in hyps)
    unk = sum(h.count("<unk>") for h in hyps)

    return unk / max(1, total)


# Calculate BLEU and ROUGE-L
def score(hyps, refs):

    bleu = sacrebleu.corpus_bleu(
        hyps,
        [refs]
    ).score

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
        tokenizer=WhitespaceTokenizer()
    )

    rouge = sum(
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hyps, refs)
    ) / len(refs)

    return bleu, rouge, unk_rate(hyps)


# Calculate perplexity using teacher-forced loss
def calculate_perplexity(pairs):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for src, tgt in pairs:
            src_ids = sp.encode_as_ids(src)
            tgt_ids = [sp.bos_id()] + sp.encode_as_ids(tgt) + [sp.eos_id()]

            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)

            mask = (src_tensor != 0).to(device)
            enc_out, h, c = enc(src_tensor)

            outputs, _ = dec(enc_out, h, c, tgt_tensor, teacher_forcing_ratio=0, mask=mask)

            L = tgt_tensor[:, 1:].reshape(-1)
            f_out = outputs.reshape(-1, outputs.shape[-1])

            loss = loss_fn(f_out, L)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    return torch.exp(torch.tensor(avg_loss)).item()


# Evaluate one dataset
def run(tsv_path, name):

    pairs = load_pairs(tsv_path)

    greedy = []
    beam = []
    references = []

    for src, tgt in pairs:

        greedy.append(
            greedy_decode(src)
        )

        beam.append(
            beam_search(
                src,
                beam_width=3
            )
        )

        references.append(tgt)

    g_bleu, g_rouge, g_unk = score(
        greedy,
        references
    )

    b_bleu, b_rouge, b_unk = score(
        beam,
        references
    )

    ppl = calculate_perplexity(pairs)

    print("\n" + "=" * 50)
    print(name)

    print(
        f"Greedy: BLEU={g_bleu:.2f}, "
        f"ROUGE-L={g_rouge:.4f}, "
        f"PPL={ppl:.2f}, "
        f"UNK={g_unk * 100:.2f}%"
    )

    print(
        f"Beam:   BLEU={b_bleu:.2f}, "
        f"ROUGE-L={b_rouge:.4f}, "
        f"PPL={ppl:.2f}, "
        f"UNK={b_unk * 100:.2f}%"
    )


# Run evaluation
if __name__ == "__main__":

    run(
        "data/valid.tsv",
        "UQA Validation"
    )

    # Wiki-UQA — uncomment once wiki_valid.tsv is prepared
    # run(
    #     "data/wiki_valid.tsv",
    #     "Wiki-UQA"
    # )