import torch
import torch.nn as nn

class encoder (nn.Module):
    def __init__(self,v_size,e_size,h_size,layers,dout):
        super().__init__()
        self.embedding=nn.Embedding(v_size,e_size)
        self.rnn=nn.LSTM(e_size,h_size,layers,bidirectional=True,dropout=dout,batch_first=True)
        self.hidden=nn.Linear(h_size*2,h_size)
        self.cell=nn.Linear(h_size*2,h_size)
    
    def forward(self,s):
        Embeddings=self.embedding(s)
        outputs,(hidden,cell)=self.rnn(Embeddings)
        hiddenStates=hidden[0::2,:,:]
        BhiddenStates=hidden[1::2,:,:]
        combining_hidden_states=torch.cat([hiddenStates,BhiddenStates],dim=2)
        output_hidden=self.hidden(combining_hidden_states)
        cellStates=cell[0::2,:,:]
        BcellStates=cell[1::2,:,:]
        combining_cell_states=torch.cat([cellStates,BcellStates],dim=2)
        output_cell=self.cell(combining_cell_states)

        return outputs,output_hidden,output_cell



enc = encoder(v_size=8000, e_size=256, h_size=512, layers=2, dout=0.3)
dummy = torch.randint(0, 8000, (64, 30))
outputs, hidden, cell = enc(dummy)
print("outputs:", outputs.shape)
print("hidden:", hidden.shape)
print("cell:", cell.shape)
