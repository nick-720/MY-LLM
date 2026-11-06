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
        # 1. switch on nn.Module's machinery
        super().__init__()
        # 2. make an empty (out_features, in_features) tensor on the given device and dtype
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        # 3. compute std from the formula
        std = (2 / (in_features + out_features)) ** 0.5
        # 4. fill the tensor with trunc_normal_, cutoffs at ±3 std
        nn.init.trunc_normal_(W, mean=0.0, std=std, a= -3 * std, b= 3 * std)
        # 5. wrap it in nn.Parameter and store it as self.W
        self.W = nn.Parameter(W)
        

    def forward(self, x):
        # 6. your einsum line, using self.W
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
