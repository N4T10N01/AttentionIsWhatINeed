import os
import torch
from torch.autograd import Variable
from collections import Counter
import math
from BPE_tokenizer import BPETokenizer
from make_model import make_model, subsequent_mask
from training_loop2 import  BOS_WORD, EOS_WORD, BLANK_WORD
import re

class TranslationVocab:
    def __init__(self, vocab):
        self.itos = list(dict.fromkeys(vocab))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol=None):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if eos_symbol is not None and next_word == eos_symbol:
            break
    return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol,
                      beam_size=4, alpha=0.6, eos_symbol=None):

    memory = model.encode(src, src_mask)

    # (sequence, log_prob, finished_flag)
    beams = [(torch.ones(1,1).fill_(start_symbol).type_as(src), 0.0, False)]

    def length_penalty(length):
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

    for _ in range(max_len - 1):

        new_beams = []

        all_finished = True

        for seq, score, finished in beams:

            # --- DO NOT expand finished beams ---
            if finished:
                new_beams.append((seq, score, True))
                continue

            all_finished = False

            out = model.decode(
                memory,
                src_mask,
                Variable(seq),
                Variable(subsequent_mask(seq.size(1)).type_as(src))
            )

            prob = model.generator(out[:, -1])
            topk_log_probs, topk_ids = torch.topk(prob, beam_size)
            # probs = torch.exp(prob)
            # print(probs[0, eos_symbol])

            for k in range(beam_size):
                next_word = topk_ids[0][k].item()
                next_score = score + topk_log_probs[0][k].item()

                new_seq = torch.cat([
                    seq,
                    torch.ones(1,1).type_as(src).fill_(next_word)
                ], dim=1)

                # mark as finished if EOS generated
                is_finished = (eos_symbol is not None and next_word == eos_symbol)

                new_beams.append((new_seq, next_score, is_finished))

        # --- early stopping if all beams finished ---
        if all_finished:
            break

        # --- rank beams with length penalty ---
        new_beams = sorted(
            new_beams,
            key=lambda x: x[1] / length_penalty(x[0].size(1)),
            reverse=True
        )

        beams = new_beams[:beam_size]

    # return best sequence
    return beams[0][0]

def translate_sentence(model, sentence, src_vocab, tgt_vocab, bpe_tokenizer,
                       max_len=50, beam_size=5, alpha=0.6, device=None):
    
    # --- Tokenize sentence and convert to indices ---
    tokens = bpe_tokenizer.tokenize(sentence)
    src_indices = [src_vocab.stoi[tok] for tok in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, seq_len]
    src_mask = torch.ones(1, 1, src_tensor.size(1)).type_as(src_tensor)
    
    # Start symbol
    start_symbol = tgt_vocab.stoi[BOS_WORD]
    eos_symbol = tgt_vocab.stoi[EOS_WORD]
    
    # --- Call your existing beam search ---
    translated_indices = beam_search_decode(model, src_tensor, src_mask,
                                               max_len=max_len,
                                               start_symbol=start_symbol,
                                               beam_size=beam_size,
                                               alpha=alpha,
                                               eos_symbol=eos_symbol)
    # translated_indices = greedy_decode(model, src_tensor, src_mask,
    #                                   max_len=max_len, start_symbol=start_symbol, eos_symbol=eos_symbol)
    
    # --- Convert indices to tokens and form string ---
    translated_tokens = [tgt_vocab.itos[idx.item()] for idx in translated_indices[0]]  # batch=0

    
    return translated_tokens


def bleu_score(references, hypotheses, max_n=4):

    weights = [0.25]*max_n

    clipped_counts = [0]*max_n
    total_counts = [0]*max_n

    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(references, hypotheses):

        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for n in range(1, max_n+1):

            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+n])
                for i in range(len(ref_tokens)-n+1)
            )

            hyp_ngrams = Counter(
                tuple(hyp_tokens[i:i+n])
                for i in range(len(hyp_tokens)-n+1)
            )

            total_counts[n-1] += sum(hyp_ngrams.values())

            for ng in hyp_ngrams:
                clipped_counts[n-1] += min(
                    hyp_ngrams[ng],
                    ref_ngrams.get(ng,0)
                )

    precisions = []

    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_counts[i]/total_counts[i])

    if min(precisions) == 0:
        return 0

    score = sum(w*math.log(p) for w,p in zip(weights,precisions))

    bp = 1 if hyp_length > ref_length else math.exp(1-ref_length/hyp_length)

    return bp * math.exp(score)


# --- Moses 13a-style tokenizer (approximation) ---
def moses_tokenize_13a(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    # Separate punctuation
    text = re.sub(r"([.,!?;:@#\$%&\(\)\[\]\{\}<>\"'])", r" \1 ", text)

    # Separate dashes
    text = re.sub(r"(-)", r" \1 ", text)

    # Normalize spaces again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --- BLEU evaluation directly from files ---
def compute_bleu_wmt14(model,
                       src_file,
                       tgt_file,
                       src_vocab,
                       tgt_vocab,
                       bpe_tokenizer,
                       device):

    hypotheses = []
    references = []

    model.eval()

    with torch.no_grad():
        with open(src_file, encoding="utf-8") as src_f, \
             open(tgt_file, encoding="utf-8") as tgt_f:

            for src_line, tgt_line in zip(src_f, tgt_f):

                src_sentence = src_line.strip()
                ref_sentence = tgt_line.strip()

                pred_sentence = translate_sentence(
                    model,
                    src_sentence,
                    src_vocab,
                    tgt_vocab,
                    bpe_tokenizer,
                    device=device
                )
                pred_sentence = bpe_tokenizer.detokenize(pred_sentence)
                # --- Apply Moses tokenization to BOTH sides ---
                pred_tok = moses_tokenize_13a(pred_sentence)
                ref_tok  = moses_tokenize_13a(ref_sentence)

                hypotheses.append(pred_tok)
                references.append(ref_tok)

    # --- Compute BLEU using your existing function ---
    bleu = bleu_score(references, hypotheses)

    return bleu

if __name__ == "__main__":
    # --- Usage ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cwd = os.getcwd() 
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.load(f"{cwd}/wmt_ruleset.json")
    vocab = TranslationVocab(bpe_tokenizer.vocab)
    vocab_size = len(vocab)
    pad_idx = bpe_tokenizer.vocab.index(BLANK_WORD)
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    checkpoint = torch.load(f"{cwd}/checkpoints42/epoch_19.pt", map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(checkpoint['val_loss'])
    model.eval()

    bleu = compute_bleu_wmt14(
        model,
        f"{cwd}/wmt14/wmt14.test.en",
        f"{cwd}/wmt14/wmt14.test.de",
        vocab,
        vocab,
        bpe_tokenizer,
        device
    )

    print(f"BLEU score (WMT14 de->en): {bleu:.4f}")