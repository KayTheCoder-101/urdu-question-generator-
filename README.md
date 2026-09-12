# Urdu Question Generation — From Scratch Seq2Seq with Attention

Generative AI, Fall 2026 — Assignment 01

A sequence-to-sequence model that reads an Urdu sentence with an answer marked inside it (`<ans>...</ans>`) and generates the question that answer responds to. 
Built from scratch: SentencePiece tokenizer, bidirectional LSTM encoder, Bahdanau attention, LSTM decoder.
No pretrained weights, no Transformers.

## Team

- Member 1 — [Kulsoom] 
- Member 2 — [Mobeen] 

## Setup

```bash
pip install -r requirements.txt
```

---

## Evaluation

### Automatic metrics

Reported on UQA validation and Wiki-UQA, for greedy and beam decoding:

- **BLEU-4** — sacrebleu, corpus level, on detokenised output
- **ROUGE-L** — F-measure, rouge-score package
- **Perplexity** — exp(mean validation cross-entropy) of the best checkpoint
- **`<unk>` rate** — fraction of generated tokens that are `<unk>`

*Expected range: Du et al. (2017) reached BLEU-4 ≈ 12 on English SQuAD. Roughly 6–13 expected on UQA validation, lower on Wiki-UQA.*

### Human evaluation

50 validation outputs (fixed seed), rated independently by both members, yes/no:

- **Fluency** — grammatical, natural Urdu
- **Relevance** — about the content of the source sentence
- **Answerability** — the marked span answers the generated question

---

## Results

### Table 1 — Dataset statistics

|  | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124745 | 16824 | |
| Answerable rows | 83018 | | |
| Pairs after length filter | | | |
| Mean source length | | | |
| Mean target length | | | |

### Table 2 — Model configuration

| | |
|---|---|
| Encoder / decoder type (LSTM or GRU) | |
| Layers | |
| Embedding size | |
| Hidden size | |
| Vocabulary size | |
| Trainable parameters | |
| Optimiser | |
| Learning rate | |
| Schedule | |
| Batch size | |
| Epochs | |
| Wall-clock time | |
| GPU | |

### Table 3 — Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | | | | |
| UQA valid | beam (k=3) | | | | |
| Wiki-UQA | greedy | | | | |
| Wiki-UQA | beam (k=3) | | | | |

### Table 4 — Human evaluation (50 samples)

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | | | |
| Member 2 (% yes) | | | |
| Cohen's κ | | | |


---

## Links

- Medium blog: [link]
- LinkedIn post: [link]
- GitHub repo: [link]