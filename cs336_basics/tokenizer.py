import pickle
import regex as re
from cs336_basics.train_bpe import PAT

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.cache = {}
        self.merge_ranks = {}
        for position, merge in enumerate(self.merges):
            self.merge_ranks[tuple(merge)] = position


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

    def encode(self, text:str) -> list[int]:
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        if special_tokens_sorted:
            pattern = "|".join(list(re.escape(token) for token in special_tokens_sorted))
            pieces = re.split(f"({pattern})", text)
        else:
            pieces = [text]
        ids = []
        for piece in pieces:
            if piece in special_tokens_sorted:
                ids.append(self.flipped_vocab[piece.encode('utf-8')])
            else:
                for match in PAT.finditer(piece):
                    word = match.group()
                    returned_ids = self._encode_word(word)
                    ids.extend(returned_ids)
        return ids

    def encode_iterable(self, iterable):
        for line in iterable:
            ids = self.encode(line)
            for _id in ids:
                yield _id

    def _encode_word(self, word: str) -> list[int]:
        if word in self.cache:
            return self.cache[word]
        word_bytes = word.encode("utf-8")
        bytes_list = []
        for bytes_value in word_bytes:
            bytes_list.append(bytes([bytes_value]))
        bytes_tuple = tuple(bytes_list)

        # for merge in self.merges:
        #     bytes_tuple = self._merge_pair(bytes_tuple, merge)
        while True:
            pairs = []
            for i in range(len(bytes_tuple) - 1):
                pairs.append((bytes_tuple[i], bytes_tuple[i + 1]))
            best_pair = None
            best_rank = None
            for pair in pairs:
                if pair not in self.merge_ranks:
                    continue
                rank = self.merge_ranks[pair]
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            bytes_tuple = self._merge_pair(bytes_tuple, best_pair)

        vocab_ids = []
        for token in bytes_tuple:
            vocab_ids.append(self.flipped_vocab[token])
        self.cache[word] = vocab_ids
        return vocab_ids

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

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for token_id in ids:
            bytes_list.append(self.vocab[token_id])
        text = b''.join(bytes_list).decode("utf-8", errors='replace')
        return text