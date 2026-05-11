import torch
import torch.nn.functional as F
import math


    
def multi_head_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_o: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    B, T, d = K.shape

    d_k = d // n_heads

    Q = Q.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3) # B, n_heads, T, d_k
    K = K.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    V = V.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3)

    scores = Q @ K.permute(0, 1, 3, 2) / math.sqrt(d_k)

    mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))

    attention_probs = F.softmax(scores, dim=-1)
    attention = attention_probs @ V

    attention = attention.permute(0, 2, 1, 3).reshape(B, T, d) @ W_o

    return attention

    