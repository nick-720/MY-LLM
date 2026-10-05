from cs336_basics.train_bpe import train_bpe

import pickle
import time

if __name__ == "__main__":

    start = time.time()
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    end = time.time()
    print(end - start)

    with open("tokenizers/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tokenizers/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)



