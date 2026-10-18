import numpy as np
from cs336_basics.tokenizer import Tokenizer

tok1 = Tokenizer.from_files(vocab_filepath="./tokenizers/tinystories_vocab.pkl", merges_filepath="./tokenizers/tinystories_merges.pkl", special_tokens=["<|endoftext|>"])
# tok2 = Tokenizer.from_files(vocab_filepath="./tokenizers/owt_vocab.pkl", merges_filepath="./tokenizers/owt_merges.pkl", special_tokens=["<|endoftext|>"])

with open("./data/TinyStoriesV2-GPT4-valid.txt", "r") as f, open("./data/tinystories_valid.bin", "wb") as out:
    batch = []

    for token_id in tok1.encode_iterable(f):
        batch.append(token_id)
        if len(batch) == 10_000_000:
            arr = np.array(batch, dtype=np.uint16)
            arr.tofile(out)
            batch = []
    if len(batch) > 0:
        arr = np.array(batch, dtype=np.uint16)
        arr.tofile(out)