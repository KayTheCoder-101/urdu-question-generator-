#%%
import torch
import sentencepiece as spm

from model import encoder, decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor(
    model_file="tokenizer/ur_sp.model"
)


v_size = 8000
e_size = 256
h_size = 512
layers = 2
dout = 0.3

enc = encoder(v_size, e_size, h_size, layers, dout)
dec = decoder(h_size, v_size, e_size, layers, dout)



checkpoint = torch.load(
    "best_model.pt",
    map_location=device
)

enc.load_state_dict(checkpoint["encoder"])
dec.load_state_dict(checkpoint["decoder"])


enc = enc.to(device)
dec = dec.to(device)

enc.eval()
dec.eval()



context = "نارمن (Norman: Nourmands؛ French: Normands؛ Latin: Normanni) وہ لوگ تھے جنہوں نے 10 ویں اور 11 ویں صدیوں میں <ans> فرانس </ans> کے ایک خطے نارمنڈی کو اپنا نام دیا۔"


# Convert context into token IDs
src_ids = sp.encode_as_ids(context)

src = torch.tensor(
    src_ids,
    dtype=torch.long
).unsqueeze(0).to(device)



with torch.no_grad():

    enc_out, h, c = enc(src)

    # Start decoder with BOS
    current_token = torch.tensor(
        [[sp.bos_id()]],
        dtype=torch.long
    ).to(device)

    generated_ids = []

    # Maximum question length
    for _ in range(30):

        logits, h, c, attention = dec.forward_step(
            current_token,
            h,
            c,
            enc_out
        )

        # Pick token with highest probability
        next_token = logits.argmax(-1).item()

        # Stop if EOS is generated
        if next_token == sp.eos_id():
            break

        generated_ids.append(next_token)

        # Feed predicted token back into decoder
        current_token = torch.tensor(
            [[next_token]],
            dtype=torch.long
        ).to(device)



question = sp.decode(generated_ids)

print("Context:")
print(context)

print("\nGenerated Question:")
print(question)