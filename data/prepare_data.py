# Person A — Task 1: Data preparation
# Loads uqa/UQA, extracts sentence containing each answer,
# wraps answer in <ans></ans>, filters by length, writes train.tsv/valid.tsv

from datasets import load_dataset

def load_data():
    ds = load_dataset("uqa/UQA")
    return ds

if __name__ == "__main__":
    ds = load_data()
    print(ds)
