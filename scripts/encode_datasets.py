import numpy as np
from cs336_basics.tokenizer import Tokenizer
from time import perf_counter

tok1 = Tokenizer.from_files(vocab_filepath="./tokenizers/tinystories_vocab.pkl", merges_filepath="./tokenizers/tinystories_merges.pkl", special_tokens=["<|endoftext|>"])
tok2 = Tokenizer.from_files(vocab_filepath="./tokenizers/owt_vocab.pkl", merges_filepath="./tokenizers/owt_merges.pkl", special_tokens=["<|endoftext|>"])

def encode_file(input_path, output_path, tokenizer):
    start_time = perf_counter()
    with open(input_path, "r") as f, open(output_path, "wb") as out:
        batch = []
        for token_id in tokenizer.encode_iterable(f):
            batch.append(token_id)
            if len(batch) == 10_000_000:
                arr = np.array(batch, dtype=np.uint16)
                arr.tofile(out)
                batch = []
        if len(batch) > 0:
            arr = np.array(batch, dtype=np.uint16)
            arr.tofile(out)
    end_time = perf_counter()
    print(f"{output_path}: {end_time - start_time:.0f} s")

encode_file("./data/TinyStoriesV2-GPT4-valid.txt", "./data/tinystories_valid.bin", tok1)

encode_file("./data/TinyStoriesV2-GPT4-train.txt", "./data/tinystories_train.bin", tok1)

encode_file("./data/owt_valid.txt", "./data/owt_valid.bin", tok2)

encode_file("./data/owt_train.txt", "./data/owt_train.bin", tok2)
