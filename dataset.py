import sentencepiece as spm
import torch
from torch.utils.data import Dataset
import csv
from torch.nn.utils.rnn import pad_sequence

#load the tokennizer
sp=spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")
class QGDataset(Dataset):
    def __init__(self,tsv,tokenizer):
        self.sp=spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")
        self.pairs=[]
        with open(tsv,encoding="utf-8") as f:
            r=csv.reader(f,delimiter='\t')
            for row in r:
                self.pairs.append(row)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        i=self.pairs[index]
        source,target=i
        source_id=self.sp.encode_as_ids(source)
        target_id=self.sp.encode_as_ids(target)
        return torch.tensor(source_id),torch.tensor(target_id)
def padding(tensors):
    sources=[]
    targets=[]
    for s,t in tensors:
        sources.append(s)
        targets.append(t)
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_sources,padded_targets
if __name__=="__main__":
    ds=QGDataset("data/train.tsv","tokenizer/ur_sp.model")
    print("total pairs",len(ds))
    print("1st pairs",ds.pairs[0])
    src,tgt=ds[0]
    print("source and target tensor",src,tgt)
    print(src.shape)
    print(tgt.shape)
    batch = [ds[0], ds[1], ds[2]]
    padded_src, padded_tgt = padding(batch)
    print("Padded source shape:", padded_src.shape)
    print("Padded target shape:", padded_tgt.shape)

