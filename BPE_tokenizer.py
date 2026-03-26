import json
import re
from collections import defaultdict, Counter


class BPETokenizer:

    def __init__(self, vocab_size=37000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = []
        self.word_freqs = None
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def pre_tokenize(self, text):
        return self.pattern.findall(text)


    def build_word_freqs(self, corpus):

        word_freqs = Counter()

        for sentence in corpus:
            for w in self.pre_tokenize(sentence):
                word_freqs[w] += 1

        self.word_freqs = word_freqs


    def initialize_splits(self):

        splits = {}
        pair_freqs = defaultdict(int)
        pair_to_words = defaultdict(set)

        for word, freq in self.word_freqs.items():

            split = list(word) + ["</w>"]
            splits[word] = split

            for i in range(len(split) - 1):

                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word)

        return splits, pair_freqs, pair_to_words



    def merge_pair(self, pair, splits, pair_freqs, pair_to_words):

        a, b = pair
        new_symbol = a + b

        affected_words = pair_to_words[pair]

        for word in list(affected_words):

            split = splits[word]
            freq = self.word_freqs[word]

            i = 0
            new_split = []

            while i < len(split):

                if i < len(split)-1 and split[i] == a and split[i+1] == b:

                    if i > 0:
                        prev_pair = (split[i-1], split[i])
                        pair_freqs[prev_pair] -= freq
                        pair_to_words[prev_pair].discard(word)

                    if i < len(split)-2:
                        next_pair = (split[i+1], split[i+2])
                        pair_freqs[next_pair] -= freq
                        pair_to_words[next_pair].discard(word)

                    new_split.append(new_symbol)
                    i += 2

                else:
                    new_split.append(split[i])
                    i += 1

            splits[word] = new_split

            for j in range(len(new_split)-1):
                p = (new_split[j], new_split[j+1])
                pair_freqs[p] += freq
                pair_to_words[p].add(word)

        pair_freqs[pair] = 0
        pair_to_words[pair].clear()


    def train(self, corpus):

        print("Building word frequencies...")
        self.build_word_freqs(corpus)

        splits, pair_freqs, pair_to_words = self.initialize_splits()

        alphabet = set()
        for word in self.word_freqs:
            alphabet.update(word)

        self.vocab = ["<blank>", "<s>", "</s>"] + sorted(alphabet) + ["</w>"]

        print("Training BPE...")

        while len(self.vocab) < self.vocab_size:
            if (len(self.vocab)%1000==0):
                print(f"{len(self.vocab)}/{self.vocab_size}")
            best = max(pair_freqs, key=pair_freqs.get)

            if pair_freqs[best] == 0:
                break

            self.merges.append(best)
            self.vocab.append(best[0] + best[1])

            self.merge_pair(best, splits, pair_freqs, pair_to_words)

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        print("BPE training finished")
        print("Vocab size:", len(self.vocab))



    def tokenize_word(self, word):

        word = list(word) + ["</w>"]

        pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}

        while True:

            candidate = None
            best_rank = float("inf")

            for p in pairs:
                if p in self.merge_ranks and self.merge_ranks[p] < best_rank:
                    best_rank = self.merge_ranks[p]
                    candidate = p

            if candidate is None:
                break

            a, b = candidate
            new_token = a + b

            new_word = []
            i = 0

            while i < len(word):

                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}

        return word


    def tokenize(self, text):

        words = self.pre_tokenize(text)
        tokens = []

        for word in words:

            split = self.tokenize_word(word)

            split = [s.replace("</w>", "") for s in split]

            for i, token in enumerate(split):

                if i < len(split) - 1:
                    tokens.append(token + "@@")
                else:
                    tokens.append(token)

        return tokens


    def detokenize(self, tokens):

        sentence = " ".join(tokens)
        sentence = sentence.replace("@@ ", "")
        return sentence


    def save(self, path):

        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": self.merges
            }, f)


    def load(self, path):

        with open(path) as f:
            tok = json.load(f)

        self.vocab = tok["vocab"]
        self.merges = [tuple(m) for m in tok["merges"]]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
