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

    while len(vocab) < vocab_size:

        pair_counts = {}
        for pre_token in pre_token_counts:

            freq = pre_token_counts[pre_token]

            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq

        best = None
        best_count = 0
        
        if len(pair_counts) == 0:
            break

        for pair in pair_counts:
            count = pair_counts[pair]
            if count > best_count or (count == best_count and (pair > best)) :
                best_count = count
                best = pair

        merges.append(best)

        vocab[len(vocab)] = (best[0] + best[1])

        new_pre_token_counts = {}

        for pre_token in pre_token_counts:
            freq = pre_token_counts[pre_token]
            i = 0
            new_tokens = []
            while i < len(pre_token):
                if (i <= (len(pre_token) - 2) and (pre_token[i], pre_token[i + 1]) == best):
                    new_tokens.append((pre_token[i] + pre_token[i + 1]))
                    i += 2
                else:
                    new_tokens.append(pre_token[i])
                    i += 1
            merged = tuple(new_tokens)
            new_pre_token_counts[merged] = freq

        pre_token_counts = new_pre_token_counts

    return vocab, merges
