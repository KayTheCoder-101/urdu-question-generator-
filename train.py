from dataset import QGDataset
from dataset import padding
from model import encoder,decoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ds_train=QGDataset("data/train.tsv","tokenizer/ur_sp.model")
ds_valid=QGDataset("data/valid.tsv","tokenizer/ur_sp.model")

train_loader=DataLoader(ds_train,batch_size=64,shuffle=True,collate_fn=padding)
valid_loader=DataLoader(ds_valid,batch_size=64,shuffle=False,collate_fn=padding)

v_size=8000
e_size=256
h_size=512
layers=2
dout=0.3

enc=encoder(v_size,e_size,h_size,layers,dout)
dec=decoder(h_size,v_size,e_size,layers,dout)

Loss=nn.CrossEntropyLoss(ignore_index=0)

optimizer=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=0.001)

e=5

for epoch in range(e):
    print("Epoch Number:",epoch+1)
    t_loss_list=[]
    v_loss_list=[]
    enc.train()
    dec.train()
    for src_batch,tgt_batch in train_loader:
        enc_out,h,c=enc(src_batch)
        outputs,att=dec(enc_out,h,c,tgt_batch)
        L=tgt_batch[:,1:].reshape(-1)
        f_out=outputs.reshape(-1,v_size)
        T_loss=Loss(f_out,L)
        optimizer.zero_grad() #poorany grad clean woh comput hou rahy thy sath sath
        T_loss.backward()
        optimizer.step()
        t_loss_list.append(T_loss.item())
    print("avg training loss",sum(t_loss_list)/len(t_loss_list))
    enc.eval()
    dec.eval()
    with torch.no_grad(): 
        for src,tgt in valid_loader:
            enc_out,h,c=enc(src)
            outputs,att=dec(enc_out,h,c,tgt)
            L_v=tgt[:,1:].reshape(-1)
            for_out=outputs.reshape(-1,v_size)
            V_loss=Loss(for_out,L_v)
            v_loss_list.append(V_loss.item())
        print("avg validation loss",sum(v_loss_list)/len(v_loss_list))
