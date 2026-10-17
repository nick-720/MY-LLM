from cs336_basics.tokenizer import Tokenizer

tok1 = Tokenizer.from_files(vocab_filepath="./tokenizers/tinystories_vocab.pkl", merges_filepath="./tokenizers/tinystories_merges.pkl", special_tokens=["<|endoftext|>"])
tok2 = Tokenizer.from_files(vocab_filepath="./tokenizers/owt_vocab.pkl", merges_filepath="./tokenizers/owt_merges.pkl", special_tokens=["<|endoftext|>"])

def compression_ratio(data_path, tokenizer):
    with open(data_path, "r") as f:
        text = f.read(1_000_000)
        docs = text.split("<|endoftext|>")
        docs = docs[1:11]
    print(len(docs))
    print(docs[0])

    bytes_total = 0
    tokens = 0
    for doc in docs:
        bytes_in_doc = len(doc.encode("utf-8"))
        bytes_total += bytes_in_doc
        tokens_in_doc = len(tokenizer.encode(doc))
        tokens += tokens_in_doc

    return (bytes_total / tokens)

ratio_tinystories = compression_ratio("./data/TinyStoriesV2-GPT4-valid.txt", tok1)
ratio_owt = compression_ratio("./data/owt_valid.txt", tok2)
ratio_owt_through_ts = compression_ratio("./data/owt_valid.txt", tok1)
print(f"tiny stories ratio: {ratio_tinystories}")
print(f"owt ratio: {ratio_owt}")
print(f"owt / ts ratio: {ratio_owt_through_ts}")