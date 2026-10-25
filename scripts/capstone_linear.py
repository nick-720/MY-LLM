from einops import einsum
import torch

x = torch.randn(3, 5, 6) # (batch, seq, d_in)
W = torch.randn(4, 6) # (d_out, d_in)

out = einsum(x, W, "batch seq d_in, d_out d_in -> batch seq d_out")

W2 = W.T 

out_matmul = x @ W2

print(out.shape)
print(out_matmul.shape)
print(torch.allclose(out, out_matmul))

