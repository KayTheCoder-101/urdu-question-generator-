# Urdu Question Generation

Sequence-to-sequence question generation for Urdu, built and trained from scratch using an RNN encoder-decoder with Bahdanau attention. Given an Urdu sentence with an answer span marked inside `<ans>...</ans>` tags, the model generates the question that answer responds to.

## Team

- **Person A:** Kulsoom — Encoder, training loop, data prep
- **Person B:** Mobeen — Attention, Decoder, decoding, evaluation

## Overview

This is a sequence-to-sequence question generation task built entirely from scratch — no pretrained models, no Transformers, no off-the-shelf seq2seq libraries. The model is trained on the UQA dataset (Urdu translation of SQuAD 2.0) and evaluated on both UQA validation and the out-of-domain Wiki-UQA test set.

**Example:**

Input:  "دریائے سندھ کی لمبائی <ans> 3180 کلومیٹر </ans> ہے۔"
Output: "دریائے سندھ کی لمبائی کتنی ہے؟"

## Project Structure

urdu-question-generation/
├── data/
│   ├── prepare_data.py       # Loads UQA, extracts sentences, wraps answers, filters by length
│   ├── train.tsv              # Generated training pairs (gitignored, regenerate via script)
│   └── valid.tsv               # Generated validation pairs (gitignored, regenerate via script)
│
├── tokenizer/
│   ├── train_tokenizer.py     # Trains SentencePiece tokenizer on train.tsv
│   ├── ur_sp.model             # Trained SentencePiece model
│   └── ur_sp.vocab              # Vocabulary file
│
├── model.py                    # Encoder, BahdanauAttention, Decoder classes
├── dataset.py                    # QGDataset (tokenization) + collate_fn (padding)
├── train.py                       # Training loop: loss, optimizer, checkpointing, early stopping
├── decode.py                       # greedy_decode() and beam_search() functions
├── evaluate.py                      # BLEU-4, ROUGE-L, Perplexity, unk-rate on UQA/Wiki-UQA
├── app.py                            # Frontend (Streamlit/Gradio) for interactive demo
│
├── results/
│   ├── length_histograms.png          # Source/target length distributions
│   ├── loss.png                        # Training vs validation loss curve
│   ├── attention_heatmap.png            # Attention visualization for a sample decode
│   ├── samples.tsv                       # 50 validation examples: source, reference, greedy, beam
│   ├── automatic_metrics.csv              # BLEU/ROUGE/PPL/unk% per split and decoding method
│   └── screenshot.png                      # Frontend UI screenshot
│
├── notebook.ipynb                # Colab driver notebook (clone, install, train, evaluate)
├── medium_blog_draft.md           # Blog post draft (problem, design, results, failure cases)
├── best_model.pt                   # Trained model checkpoint (gitignored — see Model Checkpoint below)
├── requirements.txt
├── .gitignore
└── README.md

## Model Architecture

| Component | Detail |
|---|---|
| Encoder | 2-layer bidirectional LSTM |
| Decoder | 2-layer LSTM with Bahdanau attention (attention masked against padding) |
| Bridging | Concatenation of forward/backward states + Linear projection, per layer |
| Embedding size | 256 |
| Hidden size | 512 |
| Dropout | 0.3–0.4 |
| Vocabulary | 8,000 subwords (SentencePiece, unigram model) |
| Weight tying | Output projection shares weights with the embedding layer |
| Special tokens | `<pad>`=0, `<unk>`=1, `<s>`=2, `</s>`=3, plus `<ans>`/`</ans>` as user-defined symbols |

## Training Setup

- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Loss:** Cross-entropy, padding positions ignored
- **Teacher forcing:** 0.5 ratio during training, 0 during validation
- **Gradient clipping:** max_norm=1.0
- **Early stopping:** patience=3 epochs on validation loss
- **Batch size:** 64
- **Hardware:** Google Colab (Tesla T4 GPU)

## Setup

git clone https://github.com/KayTheCoder-101/urdu-question-generator-.git
cd urdu-question-generator-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Usage

**1. Prepare the data** (downloads UQA, builds sentence-level source/target pairs):

python3 data/prepare_data.py

**2. Train the tokenizer:**

python3 tokenizer/train_tokenizer.py

**3. Train the model** (best done on GPU — see `notebook.ipynb` for Colab setup):

python3 train.py

**4. Decode a sentence** (greedy + beam search):

python3 decode.py

**5. Run full evaluation** (BLEU-4, ROUGE-L, Perplexity, unk-rate on UQA validation and Wiki-UQA):

python3 evaluate.py

**6. Launch the frontend:**

streamlit run app.py

## Results

*(to be filled in — see `results/automatic_metrics.csv` and `results/samples.tsv` once final evaluation is complete)*

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | unk% |
|---|---|---|---|---|---|
| UQA valid | greedy | — | — | — | — |
| UQA valid | beam (k=3) | — | — | — | — |
| Wiki-UQA | greedy | — | — | — | — |
| Wiki-UQA | beam (k=3) | — | — | — | — |

## Model Checkpoint

`best_model.pt` is not committed to this repository (large binary files are excluded via `.gitignore`). To reproduce it, run `train.py` following the training setup above, or contact the team for the trained checkpoint.

## Notes

- No pretrained weights, embeddings, or checkpoints were used anywhere in this project.
- All code was written from scratch by the team; AI tools were used only to explain concepts and help debug, never to generate submitted code.
- Both team members can explain every component of the pipeline.