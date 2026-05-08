import numpy as np

import numpy as np

def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray = None,
    beta: np.ndarray = None,
    eps: float = 1e-6
) -> np.ndarray:
    """
    x: (..., D)

    gamma: (D,)
    beta: (D,)
    """

    D = x.shape[-1]

    if gamma is None:
        gamma = np.ones((D,))

    if beta is None:
        beta = np.zeros((D,))

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)

    out = gamma * x_norm + beta

    return out


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
def gelu(x):
    # efficient implementation
    return 0.5 * x * (
        1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        )
    )

    
def MHSA(Q:  np.ndarray, K: np.ndarray ,V: np.ndarray, num_heads: int):
    
    B, T, D = Q.shape

    d_k = D // num_heads

    Q = np.reshape(Q, (B, T, num_heads, d_k)).transpose(0, 2, 1, 3) 
    K = np.reshape(K, (B, T, num_heads, d_k)).transpose(0, 2, 1, 3)
    V = np.reshape(V, (B, T, num_heads, d_k)).transpose(0, 2, 1, 3)

    outputs = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)) @ V


    return outputs.transpose(0, 2, 1, 3).reshape(B, T, D)
    

def mlp(x, mlp_ratio: int, embed_dim: int, W1: np.ndarray = None, W2: np.ndarray = None):


    hidden_dim = int(embed_dim * mlp_ratio)
    if W1 is None:
        W1 = np.random.randn(embed_dim, hidden_dim)

    if W2 is None:
        W2 = np.random.randn(hidden_dim, embed_dim)
        
        
    return gelu(x @ W1) @ W2
    
def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE

    x_hat = layer_norm(x)

    if Wq is None:
        Wq = np.random.randn(embed_dim, embed_dim) * 0.02

    if Wk is None:
        Wk = np.random.randn(embed_dim, embed_dim) * 0.02

    if Wv is None:
        Wv = np.random.randn(embed_dim, embed_dim) * 0.02
        
    if Wo is None:
        Wo = np.random.randn(embed_dim, embed_dim) * 0.02
        
    Q = x_hat @ Wq
    K = x_hat @ Wk
    V = x_hat @ Wv

    attn = MHSA(Q, K, V, num_heads)
    x_ = x + attn @ Wo

    output = x_ + mlp(layer_norm(x_), mlp_ratio, embed_dim, W1, W2)

    return output




    

    

    

    
