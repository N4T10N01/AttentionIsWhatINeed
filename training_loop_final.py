import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from make_model import *
from BPE_tokenizer2 import *

# === DDP CHANGE ===
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# === DDP CHANGE ===
def cleanup_distributed():
    dist.destroy_process_group()

# === DDP CHANGE ===
def is_main():
    return dist.get_rank() == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask))
        return tgt_mask
    
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:

            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, (loss.float() / batch.ntokens.float()).item(),tokens.float().item() / elapsed))
            start = time.time()
            tokens = 0
    return total_loss.float() / total_tokens.float()

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = target == self.padding_idx
        true_dist[mask] = 0.0  # directly zero out padding positions

        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, (batch, 10))  # random token indices
        src = data.long()  # cast here
        tgt = data.long()  # cast here
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        norm = norm.float()
        x = self.generator(x)

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item() * norm

from torch.utils.data import Dataset, DataLoader
from torchtext import data

# =========================================================
# === TOKENIZER + TORCHTEXT ================================
# =========================================================

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

bpe_tokenizer = BPETokenizer()
bpe_tokenizer.load("./wmt_ruleset.json")

SRC = data.Field(
    tokenize=lambda text: bpe_tokenizer.tokenize(text),
    pad_token=BLANK_WORD
)

TGT = data.Field(
    tokenize=lambda text: bpe_tokenizer.tokenize(text),
    init_token=BOS_WORD,
    eos_token=EOS_WORD,
    pad_token=BLANK_WORD
)

MAX_LEN = 120

def filter_pred(example):
    return len(example.src) <= MAX_LEN and len(example.trg) <= MAX_LEN


# =========================================================
# === DATA LOADING ========================================
# =========================================================

def load_parallel_corpus(de_path, en_path, shuffle=False, pool_size=1000, seed=42):
    examples = []
    with open(de_path, encoding="utf-8") as f_de, open(en_path, encoding="utf-8") as f_en:
        for src_line, trg_line in zip(f_de, f_en):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if src_line and trg_line:
                examples.append(
                    data.Example.fromlist(
                        [src_line, trg_line],
                        fields=[('src', SRC), ('trg', TGT)]
                    )
                )

    if shuffle:
        # --- 1. Global sort by length (stratification) ---
        examples.sort(key=lambda x: max(len(x.src), len(x.trg)))
        # --- 2. Partition into pools ---
        pools = [ examples[i:i + pool_size] for i in range(0, len(examples), pool_size) ]
        rng = random.Random(seed)
        # --- 3. Shuffle within each pool ---
        for pool in pools:
            rng.shuffle(pool)
        # --- 4. Shuffle pool order (prevents curriculum bias) ---
        rng.shuffle(pools)
        # --- 5. Flatten back to dataset ---
        examples = [ex for pool in pools for ex in pool]

    return examples

# =========================================================
# === VOCAB ===============================================
# =========================================================

class SimpleVocab:
    def __init__(self, vocab):
        self.itos = list(dict.fromkeys(tok.replace("</w>", "") for tok in vocab))
        base = {tok: i for i, tok in enumerate(self.itos)}

        class NormalizedDict(dict):
            def __getitem__(self, key):
                key = key.replace("@@", "").replace("</w>", "")
                return dict.__getitem__(self, key)

        self.stoi = NormalizedDict(base)

    def __len__(self):
        return len(self.itos)


class TranslationVocab:
    def __init__(self, vocab):
        self.itos = list(dict.fromkeys(vocab))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

SRC.vocab = TranslationVocab(bpe_tokenizer.vocab)
TGT.vocab = TranslationVocab(bpe_tokenizer.vocab)

pad_idx = TGT.vocab.stoi[BLANK_WORD]


# =========================================================
# === DDP DATA WRAPPER ====================================
# =========================================================

# === DDP CHANGE ===
class TorchtextDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.examples = dataset.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



# === DDP CHANGE ===
def collate_fn(batch, pad_idx):
    # === 1. Sort by source length (descending) ===
    #batch = sorted(batch, key=lambda x: len(x.src), reverse=True)

    src_batch = []
    trg_batch = []

    for ex in batch:
        src_batch.append(ex.src)
        trg_batch.append(ex.trg)

    # === 2. Pad tokens & Numericalize ===
    src_padded = SRC.pad(src_batch)
    trg_padded = TGT.pad(trg_batch)

    src_tokens = SRC.numericalize(src_padded).squeeze(-1).transpose(0,1)
    trg_tokens = TGT.numericalize(trg_padded).squeeze(-1).transpose(0,1)

    # === 3. HARD TOKEN CAP (prevents OOM) ===
    #MAX_TOKENS = 6250  # try 2000 if still OOM

    #B, L = trg_tokens.shape

    #if B * L > MAX_TOKENS:
     #   new_B = max(1, MAX_TOKENS // L)  # ensure at least 1 example
      #  src_tokens = src_tokens[:new_B]
       # trg_tokens = trg_tokens[:new_B]

    # === 4. Return standard Batch object ===
    return Batch(src_tokens, trg_tokens, pad_idx)
# =========================================================
# === TRAIN LOOP ==========================================
# =========================================================

def train_epoch(model, loader, loss_compute, device):
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)

    for batch in loader:
        batch.src = batch.src.to(device)
        batch.trg = batch.trg.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.trg_mask = batch.trg_mask.to(device)
        batch.trg_y = batch.trg_y.to(device)

        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens

    # === DDP CHANGE === global reduction
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    return (total_loss / total_tokens).item()


def validate(model, loader, loss_compute, device):
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in loader:
            batch.src = batch.src.to(device)
            batch.trg = batch.trg.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.trg_mask = batch.trg_mask.to(device)
            batch.trg_y = batch.trg_y.to(device)

            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

            loss = loss_compute(out, batch.trg_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens

    # === DDP CHANGE ===
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    return (total_loss / total_tokens).item()


# =========================================================
# === MAIN ================================================
# =========================================================

def main():

    # === DDP CHANGE ===
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    cwd = os.getcwd()
    # tokenizer
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.load(f"{cwd}/wmt_ruleset.json")

    vocab_size = len(bpe_tokenizer.vocab)
    pad_idx = bpe_tokenizer.vocab.index("<blank>")

    # model
    model = make_model(vocab_size, vocab_size, N=6).to(device)

    # === DDP CHANGE === wrap model
    model = DDP(model, device_ids=[local_rank])

    # optimizer
    optimizer = torch.optim.Adam(
        model.module.parameters(),  # === DDP CHANGE ===
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # === DDP CHANGE === scale LR
    model_opt = NoamOpt(
        model.module.src_embed[0].d_model,
        factor=1 * dist.get_world_size(),
        warmup=4000,
        optimizer=optimizer,
    )

    criterion = LabelSmoothing(vocab_size, pad_idx).to(device)

    # === DDP CHANGE === dataset + sampler
    # === DDP CHANGE === dataset + sampler

    train_examples = load_parallel_corpus(f"{cwd}/wmt14/wmt14.train.en", f"{cwd}/wmt14/wmt14.train.de", shuffle=True)
    val_examples   = load_parallel_corpus(f"{cwd}/wmt14/wmt14.validation.en", f"{cwd}/wmt14/wmt14.validation.de")

    train = data.Dataset(train_examples, fields=[('src', SRC), ('trg', TGT)])
    val   = data.Dataset(val_examples, fields=[('src', SRC), ('trg', TGT)])

    train.examples = [ex for ex in train.examples if filter_pred(ex)]
    val.examples   = [ex for ex in val.examples if filter_pred(ex)]

    train_dataset = TorchtextDatasetWrapper(train)
    val_dataset   = TorchtextDatasetWrapper(val)

    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=113,
        sampler=train_sampler,
        collate_fn=lambda x: collate_fn(x, pad_idx)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=113,
        sampler=val_sampler,
        collate_fn=lambda x: collate_fn(x, pad_idx)
    )

    os.makedirs(f"{cwd}/checkpoints42", exist_ok=True)

    for epoch in range(20):
        # === DDP CHANGE === shuffle per epoch
        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            SimpleLossCompute(model.module.generator, criterion, model_opt),
            device,
        )
        val_loss = validate(
            model,
            val_loader,
            SimpleLossCompute(model.module.generator, criterion, None),
            device,
        )

        # === DDP CHANGE === sync before checkpoint
        dist.barrier()

        if is_main():
            print(f"Epoch {epoch} Train Loss: {train_loss}")
            print(f"Epoch {epoch} Val Loss: {val_loss}")

            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": model_opt.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, f"checkpoints42/epoch_{epoch}.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()
