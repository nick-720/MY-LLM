import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def train_bpe(input_path, vocab_size, special_tokens):

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    pattern = "|".join(list(re.escape(token) for token in special_tokens))

    pre_token_counts = {}
    word_counts = {}

    for piece in re.split(pattern, text):

        for match in PAT.finditer(piece):

            word = match.group()

            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1 


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
    for pre_token in pre_token_counts:

            freq = pre_token_counts[pre_token]

            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq

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
        words_with_best_list = []
        for pre_token in pre_token_counts:
            for i in range(len(pre_token) - 1):
                if (pre_token[i], pre_token[i + 1]) == best:
                    words_with_best_list.append(pre_token)
                    break
    
        # Fix each word on the list. 
        for word in words_with_best_list:
            freq = pre_token_counts[word]

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            
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

            del pre_token_counts[word]
            pre_token_counts[merged] = freq

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
