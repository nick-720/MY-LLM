text = "aaabdaaabac"

tokens = tuple(bytes([x]) for x in text.encode("utf-8"))

NUM_MERGES = 3

for merge_num in range(NUM_MERGES):

    counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in counts:
            counts[pair] += 1
        else: 
            counts[pair] = 1

    best = None
    best_count = 0
    for pair in counts:
        if counts[pair] > best_count:
            best_count = counts[pair]
            best = pair

    new_tokens = []
    i = 0
    while i < len(tokens):
        if (i <= (len(tokens) - 2) and (tokens[i], tokens[i + 1]) == best):
            new_tokens.append((tokens[i] + tokens[i + 1]))
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    tuple_tokens = tuple(new_tokens)

    tokens = tuple_tokens

    print(merge_num, best, tokens)