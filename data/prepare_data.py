from datasets import load_dataset

def load_data():
    ds=load_dataset("uqa/UQA")
    return ds

if __name__ == "__main__":
    ds= load_data()
    print(ds)
