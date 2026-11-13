# token IDs              (32, 256)
# → Embedding            (32, 256, 512)
# → Transformer block × N            shape stays (32, 256, 512)
#       RMSNorm → attention          positions look back at earlier ones
#       RMSNorm → feed-forward       each position on its own
# → RMSNorm
# → Linear               (32, 256, 10000)   a score for every token 

import torch
import torch.nn as nn
from einops import einsum

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