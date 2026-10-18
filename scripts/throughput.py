from cs336_basics.tokenizer import Tokenizer
from time import perf_counter
tok = Tokenizer.from_files(vocab_filepath="./tokenizers/owt_vocab.pkl", merges_filepath="./tokenizers/owt_merges.pkl", special_tokens=["<|endoftext|>"])

with open("./data/owt_valid.txt", "r") as f:
    text = f.read(10_000_000)
    bytes_total = len(text.encode("utf-8"))
    start_time = perf_counter()
    tok.encode(text)
    end_time = perf_counter()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")
    print(f"Throughput: {bytes_total / time_taken} bytes/second")
