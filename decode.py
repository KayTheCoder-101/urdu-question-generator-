import torch
import sentencepiece as spm

from model import encoder, decoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# load tokenizer

sp = spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")

# model settings

v_size = 8000
e_size = 256
h_size = 512
layers = 2
dout = 0.3


# create model

enc = encoder(
    v_size,
    e_size,
    h_size,
    layers,
    dout
)

dec = decoder(
    h_size,
    v_size,
    e_size,
    layers,
    dout
)

# load trained model

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


# greedy decoding

def greedy_decode(context, max_len=30):

    src_ids = sp.encode_as_ids(context)

    src = torch.tensor(
        src_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        enc_out, h, c = enc(src)
        mask = (src != 0).to(device)

        current_token = torch.tensor(
            [[sp.bos_id()]],
            dtype=torch.long
        ).to(device)

        generated_ids = []

        for _ in range(max_len):

            logits, h, c, attention = dec.forward_step(
                current_token,
                h,
                c,
                enc_out,
                mask
            )

            next_token = logits.argmax(-1).item()

            if next_token == sp.eos_id():
                break

            generated_ids.append(next_token)

            current_token = torch.tensor(
                [[next_token]],
                dtype=torch.long
            ).to(device)

    return sp.decode(generated_ids)


# beam search

def beam_search(context, beam_width=3, max_len=30):

    src_ids = sp.encode_as_ids(context)

    src = torch.tensor(
        src_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        enc_out, h, c = enc(src)
        mask = (src != 0).to(device)

        bos = sp.bos_id()
        eos = sp.eos_id()

        beams = [
            (
                [bos],
                h,
                c,
                0.0,
                False
            )
        ]

        for _ in range(max_len):

            candidates = []

            for tokens, beam_h, beam_c, score, finished in beams:

                if finished:
                    candidates.append(
                        (
                            tokens,
                            beam_h,
                            beam_c,
                            score,
                            True
                        )
                    )
                    continue

                current_token = torch.tensor(
                    [[tokens[-1]]],
                    dtype=torch.long
                ).to(device)

                logits, new_h, new_c, attention = dec.forward_step(
                    current_token,
                    beam_h,
                    beam_c,
                    enc_out,
                    mask
                )

                log_probs = torch.log_softmax(
                    logits,
                    dim=-1
                )

                top_log_probs, top_tokens = torch.topk(
                    log_probs,
                    beam_width
                )

                for i in range(beam_width):

                    next_token = top_tokens[0, i].item()
                    token_score = top_log_probs[0, i].item()

                    new_tokens = tokens + [next_token]
                    new_score = score + token_score

                    is_finished = (
                        next_token == eos
                    )

                    candidates.append(
                        (
                            new_tokens,
                            new_h,
                            new_c,
                            new_score,
                            is_finished
                        )
                    )

            candidates.sort(
                key=lambda x: x[3],
                reverse=True
            )

            beams = candidates[:beam_width]

            if all(
                beam[4]
                for beam in beams
            ):
                break

        best_beam = max(
            beams,
            key=lambda x: x[3]
        )

        best_tokens = best_beam[0]

        if best_tokens[0] == bos:
            best_tokens = best_tokens[1:]

        if eos in best_tokens:
            best_tokens = best_tokens[
                :best_tokens.index(eos)
            ]

        return sp.decode(best_tokens)


# test examples

test_examples = [
    "قائداعظم محمد علی جناح <ans> 1948 </ans> میں انتقال کر گئے۔",
    "علامہ اقبال کو <ans> مشرق کا شاعر </ans> کہا جاتا ہے۔",
    "کراچی پاکستان کا سب سے بڑا <ans> شہر </ans> ہے۔",
    "دریائے سندھ کی لمبائی <ans> 3180 کلومیٹر </ans> ہے۔",
    "یہ اجلاس <ans> اسلام آباد </ans> میں منعقد ہوا۔",
]


# test

if __name__ == "__main__":

    for i, context in enumerate(test_examples, 1):

        print("\n" + "=" * 60)
        print("Example", i)

        print("\nContext:")
        print(context)

        greedy_question = greedy_decode(context)

        beam_question = beam_search(
            context,
            beam_width=3,
            max_len=30
        )

        print("\nGreedy Search:")
        print(greedy_question)

        print("\nBeam Search:")
        print(beam_question)