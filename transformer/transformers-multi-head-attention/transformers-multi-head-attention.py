import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    B, seq, d_model = Q_proj.shape

    # print(f"Q_proj shape: {Q_proj.shape}")
    # print(f"K_proj shape: {K_proj.shape}")
    # print(f"V_proj shape: {V_proj.shape}")

    # print(B, seq, d_model)

    d_k = d_model // num_heads

    Q_proj = np.reshape(Q_proj, (B, seq, num_heads, d_k)).transpose(0, 2, 1, 3)
    K_proj = np.reshape(K_proj, (B, seq, num_heads, d_k)).transpose(0, 2, 1, 3)
    V_proj = np.reshape(V_proj, (B, seq, num_heads, d_k)).transpose(0, 2, 1, 3)

    # print(f"Q_proj reshaped shape: {Q_proj.shape}")
    # print(f"K_proj reshaped shape: {K_proj.shape}")
    # print(f"V_proj reshaped shape: {V_proj.shape}")
    

    attn_scores = softmax(np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)) / np.sqrt(d_k), axis=-1)

    print(f"Attention scores shape: {attn_scores.shape}")
    print(f"V_proj shape: {V_proj.shape}")

    attn = np.matmul(attn_scores, V_proj).transpose(0, 2, 1, 3).reshape(B, seq, d_model)

    print(f"Attention output shape: {attn.shape}")
    print(f"W_o shape: {W_o.shape}")

    multi_headed = np.dot(attn, W_o)
    
    

    return multi_headed
    
    
    

    