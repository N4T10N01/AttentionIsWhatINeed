import os
import torch
from collections import OrderedDict
from make_model import make_model
from BPE_tokenizer import BPETokenizer
from bleu_score import TranslationVocab
from training_loop import BLANK_WORD

def average_checkpoints(ckpt_paths, device="cpu"):
    avg_state = OrderedDict()
    n = len(ckpt_paths)

    for i, path in enumerate(ckpt_paths):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model"]

        for k, v in state_dict.items():
            v = v.float()  # ensure fp32 accumulation

            if i == 0:
                avg_state[k] = v.clone()
            else:
                avg_state[k] += v

    for k in avg_state:
        avg_state[k] /= n

    return avg_state


device = torch.device("cpu")
cwd = os.getcwd()

# Load tokenizer
bpe_tokenizer = BPETokenizer()
bpe_tokenizer.load(f"{cwd}/wmt_ruleset.json")
vocab = TranslationVocab(bpe_tokenizer.vocab)
vocab_size = len(vocab)
pad_idx = bpe_tokenizer.vocab.index(BLANK_WORD)


# usage
ckpts = [
    f"{cwd}/checkpoints44/epoch_16.pt",
    f"{cwd}/checkpoints44/epoch_17.pt",
    f"{cwd}/checkpoints44/epoch_18.pt",
    f"{cwd}/checkpoints44/epoch_19.pt",
]

avg_state_dict = average_checkpoints(ckpts)

model = make_model(vocab_size, vocab_size, N=6).to(device)
# load into model
model.load_state_dict(avg_state_dict)

# optionally save
torch.save({"model": avg_state_dict}, "averaged_model2.pt")