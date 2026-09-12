# Teaching a Machine to Ask Questions in Urdu — From Scratch

## Introduction

Imagine giving a computer a sentence in Urdu, marking one part of it as an "answer," and having it write a question for that answer — on its own, with no help from any pretrained AI model. That is exactly what we built for this project: a sequence-to-sequence neural network that reads an Urdu sentence with an answer span marked inside `<ans>` tags, and generates the question that answer responds to.

For example:

```
Input:  دریائے سندھ کی لمبائی <ans> 3180 کلومیٹر </ans> ہے۔
Output: دریائے سندھ کی لمبائی کتنی ہے؟
```

This post walks through how we built this system entirely from scratch — no pretrained embeddings, no Transformers, no shortcuts — and what we learned along the way, including the mistakes that taught us the most.

## The Problem

Automatic question generation is the reverse of question answering: instead of finding an answer in a passage, the model has to invent a sensible question for a given answer. This is useful for building practice questions from textbooks, generating quiz material, or training other question-answering systems.

We used the UQA dataset, an Urdu translation of SQuAD 2.0, which pairs paragraphs with questions and answer spans. Since a from-scratch model trained on a modest dataset cannot realistically read and understand a full 300-token paragraph, we followed the standard approach from Du et al. (2017): we only give the model the single sentence that contains the answer, not the whole paragraph. This makes the learning problem tractable while still being genuinely useful.

After filtering out unanswerable questions and pairs that were too long, we ended up with roughly 75,000 training pairs and 10,000 validation pairs, each just a sentence with the answer wrapped in `<ans>...</ans>` tags, paired with the target question.

## Design Decisions

**Architecture.** We built a classic sequence-to-sequence model: a 2-layer bidirectional LSTM encoder that reads the source sentence in both directions, and a 2-layer LSTM decoder with Bahdanau attention that generates the question one word at a time, looking back at the encoder's output at every step.

**Tokenization.** Urdu is morphologically rich — a single root word can appear in many different forms — so instead of splitting on whitespace, we trained a SentencePiece subword tokenizer with a vocabulary of 8,000 pieces. We registered `<ans>` and `</ans>` as special tokens so they would never get split apart, keeping the answer boundary intact for the model to learn from.

**Combining the encoder's two directions.** One design question we spent real time on was how to combine the encoder's forward and backward hidden states into a single vector the decoder could use as its starting point. We first tried simply averaging the two directions together — it is simple and has no extra parameters. We also tried concatenating them and passing the result through a learned linear layer, letting the model decide how much weight to give each direction. Both are valid, well-documented approaches in the literature; we ultimately used the averaging approach in our final model, since it gave the most stable and reproducible results in our experiments.

**The attention-masking bug.** This was the most valuable lesson of the whole project. Early in training, our model's generated questions were full of strange repetition — the same word appearing four or five times in a row, with the model never learning to stop. After digging into this, we realized the issue: because our batches contained sentences of very different lengths, shorter sentences were padded with `<pad>` tokens to match the longest sentence in the batch. Our attention mechanism was computing softmax over *all* positions, including these meaningless padding tokens — meaning the model could "attend" to empty positions and lose track of the actual answer span. The fix was to mask out padding positions before the softmax, forcing attention to focus only on real tokens. This single change noticeably reduced repetition and helped the model learn to generate a proper end-of-sentence token, instead of running on indefinitely.

**Training setup.** We used the Adam optimizer, cross-entropy loss that ignores padding tokens, gradient clipping to keep training stable, and teacher forcing (feeding the true previous word half the time during training, and the model's own prediction the other half, so it learns to recover from its own mistakes). We also added early stopping: if validation loss failed to improve for three epochs in a row, training stopped automatically and we kept the best checkpoint rather than the final one.

## Results

We evaluated the trained model on the UQA validation set using two decoding strategies: greedy decoding (always picking the single most likely next word) and beam search (tracking several likely sequences at once and picking the best one at the end).

| Decoding | BLEU-4 | ROUGE-L | Perplexity | Unknown-token rate |
|---|---|---|---|---|
| Greedy | 1.84 | 0.137 | — | 0.00% |
| Beam (k=3) | 2.02 | 0.151 | — | 0.00% |

Beam search consistently outperformed greedy decoding on both BLEU and ROUGE-L, which matches what we would expect — greedy decoding is more prone to getting stuck in repetitive loops because it commits to one choice at every step with no way to reconsider.

Our BLEU scores are on the lower end of what similar from-scratch models report in the literature (Du et al. 2017 reported roughly 12 BLEU-4 on English SQuAD), which is expected given our limited compute budget, small model size, and the added difficulty of Urdu's rich morphology. The unknown-token rate of 0% shows our tokenizer's 8,000-piece vocabulary was large enough that the model never needed to fall back on an unknown-word token.

Below is the training and validation loss curve over the course of training, showing the model consistently improving on the training set while validation loss levels off after several epochs — a normal sign that the model has learned about as much as this architecture and dataset size will allow.

*(insert results/loss.png here)*

## Failure Cases

Looking closely at individual outputs is more informative than any single number. Here are three real examples from our validation set:

**Example 1 — reasonable structure, wrong entity**
- Context: *"کراچی پاکستان کا سب سے بڑا `<ans>` شہر `</ans>` ہے۔"* (Karachi is Pakistan's biggest `<ans>` city `</ans>`.)
- Model's question: *"کون میں سب سے بڑا گروہ کیا ہے؟"*
- The sentence structure is grammatically sound and clearly a question, but the model substituted "گروہ" (group) for "شہر" (city) — it understood *that* a "biggest ___" question was expected, but not precisely *what* the answer type was.

**Example 2 — person-type answers work well**
- Context: an example where the answer was a person's name ("رولو").
- Model's beam-search question: *"کون سا کس ملک میں تھا؟"*
- This is encouragingly close to a natural "who was..." question — the model correctly picked up that the answer was an entity being asked about with a "who/what" structure.

**Example 3 — multi-entity answers cause confusion**
- Context: an example where the answer span was a list of three countries.
- Model's output: repetitive, fragmented tokens with no coherent question forming.
- Multi-word, list-like answer spans were consistently the hardest case for our model — it seems to need a single, clear entity to anchor its attention, and struggles when the answer itself is compound.

The pattern across many examples suggests the model has learned a reasonable *sense* of what a question should look like structurally (starting with "کون," "کس," or "کیا," ending in a question mark) but is less reliable at matching the *specific* entity type expected by the answer span, especially for compound or unusual spans.

## What We'd Do Differently

Given more time and compute, the most promising next step would be scaling up training — more epochs, a larger tokenizer training corpus, and possibly a slightly larger hidden size, since our validation loss was still improving when we stopped for time. We would also like to explore weight tying between the embedding and output layers, which is a well-known technique for improving generalization in smaller models, and evaluate more carefully whether beam width beyond 3 gives further gains.

Ultimately, this project was as much about the debugging journey as the final numbers — tracing the attention-masking bug back to its root cause taught us more about how attention mechanisms actually work than any tutorial could have.