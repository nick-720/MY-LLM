import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token_bytes

        self.flipped_vocab = {}
        for token_id, token_bytes in self.vocab.items():
            self.flipped_vocab[token_bytes] = token_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        special_tokens = special_tokens or []
        
        return cls(vocab, merges, special_tokens)