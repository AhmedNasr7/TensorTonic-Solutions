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

    
def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    B, T, D = encoder_output.shape
    
    if W_head is None:
        W_head = np.random.randn(D, num_classes)

    h_cls = encoder_output[:, 0, :]

    h_cls_norm = layer_norm(h_cls)

    logits = h_cls_norm @ W_head

    return logits
    
    
    