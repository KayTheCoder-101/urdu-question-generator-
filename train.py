from dataset import QGDataset
from dataset import padding
from model import encoder,decoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_train=QGDataset("data/train.tsv","tokenizer/ur_sp.model")
ds_valid=QGDataset("data/valid.tsv","tokenizer/ur_sp.model")

train_loader=DataLoader(ds_train,batch_size=64,shuffle=True,collate_fn=padding)
valid_loader=DataLoader(ds_valid,batch_size=64,shuffle=False,collate_fn=padding)

v_size=8000
e_size=256
h_size=512
layers=2
dout=0.4

enc=encoder(v_size,e_size,h_size,layers,dout)
dec=decoder(h_size,v_size,e_size,layers,dout)

Loss=nn.CrossEntropyLoss(ignore_index=0)

optimizer=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=0.001, weight_decay=1e-5)

enc = enc.to(device)
dec = dec.to(device)

e=15
best_val_loss=float('inf')
patience_counter=0
patience_limit=5

for epoch in range(e):
    print("Epoch Number:",epoch+1)
    t_loss_list=[]
    v_loss_list=[]
    enc.train()
    dec.train()
    for src_batch,tgt_batch in train_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        mask = (src_batch != 0).to(device)
        enc_out,h,c=enc(src_batch)
        outputs,att=dec(enc_out,h,c,tgt_batch,teacher_forcing_ratio=0.5, mask=mask)
        L=tgt_batch[:,1:].reshape(-1)
        f_out=outputs.reshape(-1,v_size)
        T_loss=Loss(f_out,L)
        optimizer.zero_grad() #poorany grad clean woh comput hou rahy thy sath sath
        T_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), max_norm=1.0)
        optimizer.step()
        t_loss_list.append(T_loss.item())
    print("avg training loss",sum(t_loss_list)/len(t_loss_list))
    enc.eval()
    dec.eval()
    with torch.no_grad(): 
        for src,tgt in valid_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            mask = (src != 0).to(device)
            enc_out,h,c=enc(src)
            outputs,att=dec(enc_out,h,c,tgt,teacher_forcing_ratio=1, mask=mask)
            L_v=tgt[:,1:].reshape(-1)
            for_out=outputs.reshape(-1,v_size)
            V_loss=Loss(for_out,L_v)
            v_loss_list.append(V_loss.item())
        avg_val_loss=sum(v_loss_list)/len(v_loss_list)
        print("avg validation loss",avg_val_loss)
        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            patience_counter=0
            torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict()}, "best_model.pt")
            print("New best model saved!")
        else:
            patience_counter+=1
            print(f"No improvement. Patience: {patience_counter}/{patience_limit}")
            if patience_counter>=patience_limit:
                print("Early stopping triggered.")
                break