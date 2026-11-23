def shares(num_layers, d_model, d_ff, seq=1024, vocab_size=50_257):
    proj = 4 * 2 * seq * d_model ** 2
    scores = 2 * (seq ** 2) * d_model
    mix = 2 * (seq ** 2) * d_model
    ffn = 3 * 2 * seq * d_model * d_ff
    block = proj + scores + mix + ffn
    lm_head = 2 * seq * d_model * vocab_size
    total = num_layers * block + lm_head
    return {"proj": num_layers * proj / total,
            "scores": num_layers * scores / total,
            "mix": num_layers * mix / total,
            "ffn": num_layers * ffn / total,
            "lm_head": lm_head / total,
            "total": total}
print(shares(48, 1600, 4288))
print(shares(12, 768, 2048))
print(shares(24, 1024, 2752))
print(shares(36, 1280, 3392))
print(shares(48, 1600, 4288, seq=16384))