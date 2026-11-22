# token IDs              (32, 256)
# → Embedding            (32, 256, 512)
# → Transformer block × N            shape stays (32, 256, 512)
#       RMSNorm → attention          positions look back at earlier ones
#       RMSNorm → feed-forward       each position on its own
# → RMSNorm
# → Linear               (32, 256, 10000)   a score for every token 

import torch
import torch.nn as nn
from einops import einsum, rearrange

def softmax(x, dim):
    x_max = x.amax(dim=dim, keepdim=True)    # 1. find the largest value along dim, kept as a size-1 dimension
    max_subtracted = x - x_max    # 2. subtract it from x
    max_subtracted_exp = torch.exp(max_subtracted)  # 3. exp every element
    result = max_subtracted_exp / max_subtracted_exp.sum(dim=dim, keepdim=True)    # 4. divide by the sum along dim, also kept as a size-1 dimension
    return result

def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")  # 1. scores = Q Kᵀ, for every batch, as one einsum
    d_k = Q.shape[-1]
    scaled_scores = scores / (d_k ** 0.5)    # 2. divide the scores by √d_k
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float("-inf"))    # 3. if there's a mask: set the scores where mask is False to −∞
    weights = softmax(scaled_scores, dim=-1)    # 4. softmax along the keys dimension
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")    # 5. multiply the weights by V and return

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()  # 1. switch on nn.Module's machinery
        W = torch.empty(out_features, in_features, device=device, dtype=dtype) # 2. make an empty (out_features, in_features) tensor on the given device and dtype
        std = (2 / (in_features + out_features)) ** 0.5 # 3. compute std from the formula
        nn.init.trunc_normal_(W, mean=0.0, std=std, a= -3 * std, b= 3 * std) # 4. fill the tensor with trunc_normal_, cutoffs at ±3 std
        self.W = nn.Parameter(W) # 5. wrap it in nn.Parameter and store it as self.W
        
    def forward(self, x):
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out") # 6. your einsum line, using self.W

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        E = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = 1
        nn.init.trunc_normal_(E, mean=0.0, std=std, a= -3 * std, b= 3 * std)
        self.E = nn.Parameter(E)

    def forward(self, token_ids):
        return self.E[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        result = (x / rms) * self.gain
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        z = self.W1(x)
        gate = z * torch.sigmoid(z)
        values = self.W3(x)
        gated_values = gate * values
        result = self.W2(gated_values)
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        speeds = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))     # 1. the speeds: one per pair, from 1 down to slow
        positions = torch.arange(max_seq_len, device=device)    # 2. the positions: 0, 1, 2, ... up to max_seq_len − 1
        angles = positions[:, None] * speeds    # 3. the angles: every position × every speed, a grid
        self.register_buffer("cos", torch.cos(angles), persistent=False)    # 4. store cos and sin of that grid, as fixed buffers
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]    # 5. look up the cos and sin rows for these tokens' positions
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]    # 6. split x into pairs: the first numbers and the second numbers
        new1 = x1 * cos - x2 * sin  
        new2 = x1 * sin + x2 * cos  # 7. rotate every pair with the recipe
        stacked = torch.stack([new1, new2], dim=-1)
        result = rearrange(stacked, "... pairs two -> ... (pairs two)")    # 8. put the pairs back together and return
        return result

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 1. store num_heads, and compute d_k
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)    # 2. four Linear layers: q_proj, k_proj, v_proj, output_proj
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)    # 3. if theta was given: a RoPE module, built with d_k
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)   # 4. make Q, K, V
        Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)      # 5. split each into heads: (..., seq, d_model) → (..., heads, seq, d_k)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions) # 6. if there's RoPE: rotate Q and K
        mask = torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)    # 7. build the causal mask
        out = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)    # 8. run your SDPA
        out = rearrange(out, "... heads seq d_k -> ... seq (heads d_k)", heads=self.num_heads) # 9. glue the heads back: (..., heads, seq, d_k) → (..., seq, d_model)
        result = self.output_proj(out)
        return result    # 10. the output layer, and return

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        # 1. build the block's four parts: ln1, attn, ln2, ffn

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)    # 2. the attention sublayer, added to the stream
        x = x + self.ffn(self.ln2(x))   # 3. the feed-forward sublayer, added to the stream
        return x    # 4. return