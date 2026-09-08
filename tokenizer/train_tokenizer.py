
from pathlib import Path
import csv
import sentencepiece as spm

# Paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

TRAIN_PATH = DATA_DIR / "train.tsv"
CORPUS_PATH = BASE_DIR / "sp_corpus.txt"
MODEL_PREFIX = BASE_DIR / "ur_sp"

# A.4 — Train the SentencePiece tokenizer

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"

#  Create tokenizer training corpus
#  Use TRAIN split only

train_pairs = []

with open(TRAIN_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(
        f,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        escapechar="\\"
    )

    for row in reader:
        if len(row) == 2:
            src, tgt = row
            train_pairs.append((src, tgt))

print(f"Loaded {len(train_pairs)} training pairs")

with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for src, tgt in train_pairs:
        f.write(src + "\n")
        f.write(tgt + "\n")

print(f"Corpus written to: {CORPUS_PATH}")

# Train SentencePiece tokenizer

spm.SentencePieceTrainer.train(
    input=str(CORPUS_PATH),
    model_prefix=str(MODEL_PREFIX),
    vocab_size=8000,
    model_type="unigram",
    character_coverage=1.0,
    user_defined_symbols=[ANS_OPEN, ANS_CLOSE],
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

# Load the trained tokenizer

sp = spm.SentencePieceProcessor(
    model_file=str(MODEL_PREFIX) + ".model"
)

PAD = 0
UNK = 1
BOS = 2
EOS = 3

# Test the tokenizer

src, tgt = train_pairs[0]

print("\nSOURCE PIECES:")
print(sp.encode(src, out_type=str))

print("\nTARGET IDS:")
print(sp.encode(tgt))

decoded = sp.decode(sp.encode(tgt))

print("\nROUND-TRIP CHECK:")
print(decoded == tgt)

