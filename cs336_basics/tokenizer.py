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

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for token_id in ids:
            bytes_list.append(self.vocab[token_id])
        text = b''.join(bytes_list).decode("utf-8", errors='replace')
        return text

    def _encode_word(self, word: str) -> list[int]:
        word_bytes = word.encode("utf-8")
        bytes_list = []
        for bytes_value in word_bytes:
            bytes_list.append(bytes([bytes_value]))
        bytes_tuple = tuple(bytes_list)
        return bytes_tuple

    def _merge_pair(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        i = 0
        new_word = []
        while i < len(word):
                if i <= (len(word) - 2) and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else: 
                    new_word.append(word[i])
                    i += 1
        tuple_word = tuple(new_word)
        return tuple_word