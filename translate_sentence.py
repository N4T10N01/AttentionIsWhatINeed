import torch
import os
from BPE_tokenizer import BPETokenizer
from training_loop2 import SimpleVocab, BOS_WORD, EOS_WORD, BLANK_WORD
from make_model import make_model
from torch.autograd import Variable
from bleu_score import TranslationVocab, translate_sentence

if __name__ == "__main__":
    device = torch.device("cpu")
    cwd = os.getcwd()

    # Load tokenizer
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.load(f"{cwd}/wmt_ruleset.json")
    vocab = TranslationVocab(bpe_tokenizer.vocab)
    vocab_size = len(vocab)
    pad_idx = bpe_tokenizer.vocab.index(BLANK_WORD)

    # Load model
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    checkpoint = torch.load(f"{cwd}/checkpoints42/epoch_19.pt", map_location=device) # Adjust path as needed
    model.load_state_dict(checkpoint['model'])
    print(checkpoint['val_loss'])
    model.eval()

    #Translate a sample sentence
    sentence = "Two sets of lights so close to one another: intentional or just a silly error?"

    print("Original sentence:", sentence)
    translation = translate_sentence(model, sentence, vocab, vocab, bpe_tokenizer)
    print("Translation:", bpe_tokenizer.detokenize(translation))
