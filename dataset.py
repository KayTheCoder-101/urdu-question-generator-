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
        target_id=[self.sp.bos_id()] + self.sp.encode_as_ids(target) + [self.sp.eos_id()]
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


