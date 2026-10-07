from cs336_basics.train_bpe import train_bpe

import pickle
import time

if __name__ == "__main__":

    start = time.time()
    vocab, merges = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"])
    end = time.time()
    print(end - start)

    with open("tokenizers/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tokenizers/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)



