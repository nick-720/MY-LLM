import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_gen(input_path, start, end, special_tokens):
    word_counts = {}
    pattern = "|".join(list(re.escape(token) for token in special_tokens))
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        for piece in re.split(pattern, chunk):
            for match in PAT.finditer(piece):
                word = match.group()
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    return word_counts

def train_bpe(input_path, vocab_size, special_tokens):

    # with open(input_path, encoding="utf-8") as f:
    #     text = f.read()

    word_counts = {}

    chunked_tuple = []

    with open(input_path, "rb") as f:
        num_processes = 32
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tup = (input_path, start, end, special_tokens)
            chunked_tuple.append(tup)
    
    with Pool(num_processes) as pool:
        chunked_word_counts = pool.starmap(chunk_gen, chunked_tuple)
    
    for chunk in chunked_word_counts:
        for word in chunk:
            if word in word_counts:
                word_counts[word] += chunk[word]
            else:
                word_counts[word] = chunk[word]
                # f.seek(start)
            # chunk = f.read(end - start).decode("utf-8")
            # for piece in re.split(pattern, chunk):
            #     for match in PAT.finditer(piece):
            #         word = match.group()
            #         if word in word_counts:
            #             word_counts[word] += 1
            #         else:
            #             word_counts[word] = 1 

    pre_token_counts = {}
    for word in word_counts:
        pre_token_counts[(tuple(bytes([x]) for x in word.encode("utf-8")))] = word_counts[word]

    # print(len(pre_token_counts))

    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[(len(vocab))] = token.encode("utf-8")

    merges = []


    # Count every pair once. Before rounds loop. We only count once, so every round uses this. 
    pair_counts = {}
    pair_to_words = {}
    for pre_token in pre_token_counts:

            freq = pre_token_counts[pre_token]

            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = {pre_token}
                else: 
                    pair_to_words[pair].add(pre_token)

# create a dict where key = pair, value = set() of words

    

    # Rounds loop
    while len(vocab) < vocab_size:

        # Pick the most common pair. 
        best = None
        best_count = 0
        
        if len(pair_counts) == 0:
            break

        for pair in pair_counts:
            count = pair_counts[pair]
            if count > best_count or (count == best_count and (pair > best)) :
                best_count = count
                best = pair
        
        # Find the words that contain best pair.
        # words_with_best_list = []
        # for pre_token in pre_token_counts:
        #     for i in range(len(pre_token) - 1):
        #         if (pre_token[i], pre_token[i + 1]) == best:
        #             words_with_best_list.append(pre_token)
        #             break
        words_with_best_list = []
        for pair in pair_to_words:
            if pair == best:
                for word in pair_to_words[pair]:
                    words_with_best_list.append(word)
    
        # Fix each word on the list. 
        for word in words_with_best_list:
            freq = pre_token_counts[word]


            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                pair_to_words[pair].discard(word)
            


            i = 0
            new_tokens = []
            while i < len(word):
                if (i <= (len(word) - 2) and (word[i], word[i + 1]) == best):
                    new_tokens.append((word[i] + word[i + 1]))
                    i += 2
                else:
                    new_tokens.append(word[i])
                    i += 1

            merged = tuple(new_tokens)
            

            for i in range(len(merged) - 1):
                pair = (merged[i], merged[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq

                if pair not in pair_to_words:
                    pair_to_words[pair] = {merged}
                else:
                    pair_to_words[pair].add(merged)


            del pre_token_counts[word]
            pre_token_counts[merged] = freq
        
        # print(pair_to_words)

        # DEBUG CHECK
        #     check_counts = {}
        #     for pre_token in pre_token_counts:
        #         freq = pre_token_counts[pre_token]
        #         for i in range(len(pre_token) - 1):
        #             pair = (pre_token[i], pre_token[i + 1])
        #             if pair in check_counts:
        #                 check_counts[pair] += freq
        #             else:
        #                 check_counts[pair] = freq

        # if check_counts != pair_counts:
        #     print(check_counts)
        #     print(pair_counts)
        # assert check_counts == pair_counts

        merges.append(best)

        vocab[len(vocab)] = (best[0] + best[1])

    return vocab, merges

# time testing script
# import time
# start = time.time()
# vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
# end = time.time()
# print(end - start)