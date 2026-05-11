import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V):
    """
    Returns: torch.Tensor of shape (batch, seq_q, d_v)
    """

    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor(Q, dtype=torch.float32)

    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32)

    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch.float32)

        
    B, seq_len, d_k = K.shape

    scores = Q @ K.permute(0, 2, 1) / math.sqrt(d_k)
    attention_probs = F.softmax(scores, dim=-1)
    attention = attention_probs @ V

    return attention

    
