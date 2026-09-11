import torch
import torch.nn as nn
import random

class encoder (nn.Module):
    def __init__(self,v_size,e_size,h_size,layers,dout):
        super().__init__()
        self.embedding=nn.Embedding(v_size,e_size, padding_idx=0)
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
    
class BahdanauAttention(nn.Module):
    def __init__(self,dec_size,enc_size):
            super().__init__()
            self.hid=nn.Linear(dec_size,enc_size)
            self.enc=nn.Linear(enc_size,enc_size)
            self.score=nn.Linear(enc_size,1)

    def forward(self,d,e):
        h=self.hid(d[-1])
        ee=self.enc(e)
        result = torch.unsqueeze(h,1)
        r=result+ee
        att=torch.tanh(r)
        s=self.score(att)
        sq=torch.squeeze(s,2)
        output=torch.softmax(sq,dim=1)
        usq=torch.unsqueeze(output,1)
        result2=torch.bmm(usq,e)
        context=result2  
        return context, output



class decoder(nn.Module):
    def __init__(self, hid_size, out_size, emb_size, layers, dout): #same dim for h_t & c_t, and for output_size & v_size
        super().__init__()
        self.embedding=nn.Embedding(out_size, emb_size, padding_idx=0)
        self.attention=BahdanauAttention(hid_size,hid_size*2)
        self.lstm=nn.LSTM(emb_size + hid_size*2, hid_size, layers, batch_first=True, dropout=dout)
        self.out=nn.Linear(hid_size, out_size)

    def forward_step(self, inp_token,dec_hid, dec_cell, enc_out): #yeh aik timestep ka kaam karta hai
        embedded = self.embedding(inp_token)
        context, weights=self.attention(dec_hid, enc_out)  
        lstm_input= torch.cat([embedded, context], dim=2)
        output, (dec_hid,dec_cell)= self.lstm(lstm_input, (dec_hid, dec_cell))  #starting mein hidden aur cell state encoder say mili hai
        logits=self.out(output.squeeze(1)) #logits abhi raw scores hain inpay softmax laga kar actual word probability milti hai
        return logits, dec_hid, dec_cell, weights

    def forward(self, enc_out, dec_hid, dec_cell, target, teacher_forcing_ratio=0.5):
        #setup
        batch_size, target_len = target.shape
        outputs = [] #predictions
        attentions = []                                    
        inp_token = target[:, 0].unsqueeze(1) #har example ki starting point <s>

        for t in range(1, target_len):
            logits, dec_hid, dec_cell, weights = self.forward_step(inp_token, dec_hid, dec_cell, enc_out)
            outputs.append(logits)
            attentions.append(weights)                     
            use_teacher_forcing = random.random() < teacher_forcing_ratio #decides random: ground truth or model's prediction
            if use_teacher_forcing:
                inp_token = target[:, t].unsqueeze(1)
            else:
                inp_token = logits.argmax(-1).unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)          
        return outputs, attentions

