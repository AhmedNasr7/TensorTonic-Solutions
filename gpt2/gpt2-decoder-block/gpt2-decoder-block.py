import torch
import math
import torch.nn.functional as F
import math


def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return x

    
def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns: torch.Tensor with LayerNorm applied across the last dimension
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    return gamma * (x - mean) / torch.sqrt(var + eps) + beta




    
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



def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor with GELU applied element-wise
    """
    return x * 0.5 * (1 + torch.erf(x / 2 ** 0.5))

    
def ffn(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of same shape as x after FFN with GELU activation
    """
    return gelu(x @ W1 + b1) @ W2 + b2

    
def gpt2_decoder_block(
    x, gamma1, beta1,
    W_q, W_k, W_v, W_o,
    gamma2, beta2,
    W1, b1, W2, b2,
    n_heads
):
    """
    Returns: nested list of shape (seq_len, d_model), rounded to 4 decimals.
    """

    x = to_tensor(x)
    gamma1 = to_tensor(gamma1)
    beta1 = to_tensor(beta1)
    W_q = to_tensor(W_q)
    W_k = to_tensor(W_k)
    W_v = to_tensor(W_v)
    W_o = to_tensor(W_o)
    gamma2 = to_tensor(gamma2)
    beta2 = to_tensor(beta2)
    W1 = to_tensor(W1)
    b1 = to_tensor(b1)
    W2 = to_tensor(W2)
    b2 = to_tensor(b2)

    squeeze_output = False

    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_output = True

    # 1) Pre-norm + causal self-attention + residual
    x_norm = layernorm(x, gamma1, beta1, eps=1e-5)

    attn_out = multi_head_attention(
        x_norm, W_q, W_k, W_v, W_o, n_heads
    )

    x = x + attn_out

    # 2) Pre-norm + FFN + residual
    x_norm = layernorm(x, gamma2, beta2, eps=1e-5)

    ffn_out = ffn(x_norm, W1, b1, W2, b2)

    x = x + ffn_out

    if squeeze_output:
        x = x.squeeze(0)

    x = torch.round(x * 10000) / 10000

    return x.tolist()