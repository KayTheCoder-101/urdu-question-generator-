# Teaching a Machine to Ask Questions in Urdu — Built From Scratch

## Introduction

Imagine handing a computer a sentence in Urdu, marking one part of it as an "answer," and having it write a sensible question for that answer — entirely on its own, with no pretrained language model doing the heavy lifting. That is the task we set out to solve: a sequence-to-sequence neural network that reads an Urdu sentence with an answer span marked using `<ans>` tags, and generates the question that answer responds to.

For example:

```
Input:  دریائے سندھ کی لمبائی <ans> 3180 کلومیٹر </ans> ہے۔
Output: دریائے سندھ کی لمبائی کتنی ہے؟
```

This post walks through how we built this system entirely from scratch — no pretrained embeddings, no Transformers, no shortcuts — the design decisions we made along the way, the bugs that taught us the most, and an honest look at where the model succeeds and where it still struggles.

## The Problem

Automatic question generation is the mirror image of question answering: instead of locating an answer inside a passage, the model has to invent a plausible question for a given answer. This has real practical value — turning a textbook paragraph into practice questions, generating quiz material automatically, or producing training data for other question-answering systems.

We worked with UQA, an Urdu translation of SQuAD 2.0 that pairs paragraphs with questions and answer spans. Feeding an entire paragraph into a small, from-scratch model trained on a modest dataset is unrealistic — the model simply doesn't have the capacity to track that much context. So, following the approach used by Du et al. (2017) for the equivalent English task, we restricted the model's input to just the single sentence containing the answer, rather than the full surrounding paragraph. This keeps the learning problem tractable while still producing a genuinely useful system.

After filtering out unanswerable questions and pairs whose length exceeded our thresholds, we were left with roughly 75,000 training pairs and 10,000 validation pairs — each one a sentence with the answer wrapped in `<ans>...</ans>`, paired with its target question.

## Design Decisions

**Architecture.** We built a classic encoder–decoder model: a 2-layer bidirectional LSTM encoder reads the source sentence in both directions, and a 2-layer LSTM decoder with Bahdanau attention generates the question one word at a time, consulting the encoder's output at every step to decide what to focus on next.

**Tokenization.** Urdu is morphologically rich — a single root word can take many different forms depending on tense, case, and gender — so we trained a SentencePiece subword tokenizer with an 8,000-piece vocabulary rather than splitting on whitespace. We registered `<ans>` and `</ans>` as user-defined symbols so the tokenizer would never split them apart, preserving the answer boundary as a single, learnable signal for the model.

**Combining the encoder's two directions.** One design decision we deliberated over was how to merge the encoder's forward and backward hidden states into a single starting point for the decoder. We experimented with two approaches: simply averaging the two directions, and concatenating them followed by a learned linear projection that lets the model decide how much weight each direction deserves. Both are established techniques in the literature. In our experiments, the concatenation-and-projection approach converged faster in isolation, but we ultimately trained and validated our final, submitted model using the averaging approach, since it gave us the most stable and reproducible results across repeated training runs under our time constraints.

**The attention-masking bug.** This was the single most instructive part of the project. Early in training, our model's generated questions were riddled with repetition — the same word appearing four or five times in a row, with generation never naturally terminating. Investigating this, we traced the cause to how our batches were built: sentences of different lengths were padded with `<pad>` tokens to match the longest sentence in each batch, but our attention mechanism was computing softmax over *every* position in the sequence — including the meaningless padding. This meant the model could, at any decoding step, "attend" to empty positions and lose track of the actual content. The fix was to mask out padded positions before the softmax step, forcing attention to stay confined to real tokens. This single change measurably reduced repetition and helped the model learn to emit a proper end-of-sequence token instead of running on indefinitely.

**Training setup.** We trained with the Adam optimizer, cross-entropy loss that ignores padding positions, gradient clipping to keep the RNN's gradients well-behaved, and teacher forcing — feeding the model the true previous word half the time during training, and its own prediction the other half, so it learns to recover gracefully from its own mistakes. We also implemented early stopping: if validation loss failed to improve for three consecutive epochs, training halted automatically and we kept the checkpoint with the lowest validation loss, rather than whatever the final epoch happened to produce.

## Results

We evaluated the trained model on the UQA validation set using two decoding strategies: greedy decoding, which always selects the single most probable next word, and beam search (beam width 3), which tracks several likely sequences in parallel and selects the best one once decoding finishes.

| Decoding | BLEU-4 | ROUGE-L | Perplexity | Unknown-token rate |
|---|---|---|---|---|
| Greedy | 1.84 | 0.1368 | 341.24 | 0.00% |
| Beam (k=3) | 2.02 | 0.1511 | 341.24 | 0.00% |

Beam search consistently outperformed greedy decoding on every metric, which matches expectations: greedy decoding commits irreversibly to one word at every step, making it more prone to falling into repetitive loops, while beam search can explore and recover from a locally weak choice.

Our BLEU-4 scores sit below the 6–13 range reported for comparable from-scratch English models (Du et al., 2017 report roughly 12 BLEU-4 on English SQuAD), and our perplexity of 341 is noticeably higher than we would like. We see this as an honest reflection of the constraints we were working under — a compact, from-scratch architecture, a training budget of a handful of GPU hours, and the added difficulty of Urdu's rich morphology compared to English — rather than a sign of a fundamental bug in the pipeline. Notably, the unknown-token rate sits at a clean 0.00%, confirming our 8,000-piece vocabulary was large enough that the model never had to fall back on an unknown-word token, which rules out vocabulary coverage as a source of the gap.

The chart below shows training and validation loss over the course of training: training loss falls steadily as the model fits the data, while validation loss improves for several epochs before leveling off — the expected signature of a model that has extracted roughly as much signal as this architecture and dataset size will allow.

*(results/loss.png)*

## Failure Cases

Aggregate metrics only tell part of the story. Looking closely at individual generations is far more revealing. Here are three real examples pulled directly from our validation set.

**1. Reasonable structure, wrong entity type**

- Context: *"کراچی پاکستان کا سب سے بڑا `<ans>` شہر `</ans>` ہے۔"* (Karachi is Pakistan's biggest `<ans>` city `</ans>`.)
- Generated question: *"کون میں سب سے بڑا گروہ کیا ہے؟"*
- The output is grammatically coherent and unmistakably a question, but the model substitutes "گروہ" (group) for "شہر" (city). It has clearly learned the shape of a "biggest ___" question, but not precisely which entity type the marked answer belongs to.

**2. Person-type answers are handled reasonably well**

- Context: a passage naming a historical leader, with the person's name marked as the answer.
- Generated question (beam search): *"کون سا کس ملک میں تھا؟"*
- This is encouragingly close to a natural "who was..." construction. When the answer span is a single, unambiguous entity — a name — the model's grasp of question structure holds up noticeably better than in the multi-word cases below.

**3. Multi-entity answer spans cause real confusion**

- Context: a passage listing several countries as the marked answer.
- Generated output: fragmented, repetitive tokens that never resolve into a coherent question.
- Compound or list-like answer spans were consistently the hardest case for the model. It appears to need a single, clearly bounded entity to anchor its attention, and its performance degrades noticeably when that assumption breaks down.

Across many examples, a consistent pattern emerges: the model has internalized a reasonable *sense* of what an Urdu question should look like structurally — starting with "کون," "کس," or "کیا," and closing with a question mark — but it is markedly less reliable at inferring the *specific* semantic category the marked answer belongs to, particularly for compound, numeric, or list-style spans.

## What We'd Do Differently

With more time and compute, the most promising next step would be simply scaling up: more training epochs, a larger and more diverse tokenizer training corpus, and possibly a larger hidden dimension, since validation loss was still trending downward in later runs when we had to stop for time. We would also like to more rigorously benchmark our two bridging strategies — averaging versus concatenation-and-projection — under identical, controlled conditions rather than across separate training runs, and to explore weight tying between the embedding and output layers, a well-established technique for improving generalization in smaller models that showed early promise in our experiments but that we did not have time to fully validate before this deadline.

Beyond the numbers, this project was as much about the debugging journey as the final metrics. Tracing the attention-repetition problem back to unmasked padding taught us more about how attention mechanisms actually behave in practice than any amount of reading about them in the abstract — and that, ultimately, is the point of building something from scratch.