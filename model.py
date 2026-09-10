import torch
import torch.nn as nn

class encoder (nn.Module):
    def __init__(self,v_size,e_size,h_size,layers,dout):
        super().__init__()
        self.embedding=nn.Embedding(v_size,e_size)
        self.rnn=nn.LSTM(e_size,h_size,layers,bidirectional=True,dropout=dout,batch_first=True)

    
    def forward(self,s):
        Embeddings=self.embedding(s)
        outputs,(hidden,cell)=self.rnn(Embeddings)
        hiddenStates=hidden[0::2,:,:]
        BhiddenStates=hidden[1::2,:,:]
        combining_hidden_states=(hiddenStates + BhiddenStates) / 2
        cellStates=cell[0::2,:,:]
        BcellStates=cell[1::2,:,:]
        combining_cell_states=(cellStates + BcellStates) / 2

        return outputs, combining_hidden_states, combining_cell_states