import torch
import torch.nn.functional as F
import math


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.double()
    return torch.tensor(x, dtype=torch.float64)


def gpt2_embedding(token_ids, token_embed_weight, position_embed_weight):
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    else:
        token_ids = token_ids.long()
    token_embed_weight = to_tensor(token_embed_weight)
    position_embed_weight = to_tensor(position_embed_weight)
    seq_len = token_ids.shape[0]
    indices = torch.arange(seq_len)
    return token_embed_weight[token_ids] + position_embed_weight[indices]


def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads):
    Q = x @ W_q.T
    K = x @ W_k.T
    V = x @ W_v.T
    B, T, d = K.shape
    d_k = d // n_heads
    Q = Q.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    K = K.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    V = V.reshape(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    scores = Q @ K.permute(0, 1, 3, 2) / math.sqrt(d_k)
    mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attention_probs = F.softmax(scores, dim=-1)
    attention = attention_probs @ V
    attention = attention.permute(0, 2, 1, 3).reshape(B, T, d) @ W_o.T
    return attention


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1.T + b1) @ W2.T + b2


def scale_residual_weights(W, N):
    W = to_tensor(W)
    return W / math.sqrt(N)


def forward_with_scaling(x, weights_list, N, use_scaling):
    x = to_tensor(x)
    for W in weights_list:
        W = to_tensor(W)
        if use_scaling:
            W = W / math.sqrt(N)
        x = x + W @ x
    return torch.norm(x, p=2).item()


def gpt2_decoder_block(x, gamma1, beta1, W_q, W_k, W_v, W_o, gamma2, beta2, W1, b1, W2, b2, n_heads):
    x = to_tensor(x)
    gamma1, beta1 = to_tensor(gamma1), to_tensor(beta1)
    W_q, W_k, W_v, W_o = to_tensor(W_q), to_tensor(W_k), to_tensor(W_v), to_tensor(W_o)
    gamma2, beta2 = to_tensor(gamma2), to_tensor(beta2)
    W1, b1, W2, b2 = to_tensor(W1), to_tensor(b1), to_tensor(W2), to_tensor(b2)

    squeeze_output = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_output = True

    x_norm = layernorm(x, gamma1, beta1)
    x = x + multi_head_attention(x_norm, W_q, W_k, W_v, W_o, n_heads)

    x_norm = layernorm(x, gamma2, beta2)
    x = x + ffn(x_norm, W1, b1, W2, b2)

    if squeeze_output:
        x = x.squeeze(0)
    return x


def gpt2_forward(token_ids, wte, wpe, layers, gamma_f, beta_f, W_lm):
    n_heads = 2
    x = gpt2_embedding(token_ids, wte, wpe)

    for layer in layers:
        x = gpt2_decoder_block(
            x, layer['gamma1'], layer['beta1'],
            layer['W_q'], layer['W_k'], layer['W_v'], layer['W_o'],
            layer['gamma2'], layer['beta2'],
            layer['W1'], layer['b1'], layer['W2'], layer['b2'], n_heads)

    gamma_f = to_tensor(gamma_f)
    beta_f = to_tensor(beta_f)
    W_lm = to_tensor(W_lm)

    logits = layernorm(x, gamma_f, beta_f) @ W_lm.T
    logits = torch.round(logits * 10000) / 10000
    return logits.tolist()