import numpy as np

def patch_embed(
    image: np.ndarray,
    patch_size: int,
    embed_dim: int,
    W_proj: np.ndarray | None = None,
) -> np.ndarray:
    B, H, W, C = image.shape

    assert H % patch_size == 0, "H must be divisible by patch_size"
    assert W % patch_size == 0, "W must be divisible by patch_size"

    n_h = H // patch_size
    n_w = W // patch_size
    N = n_h * n_w
    patch_dim = patch_size * patch_size * C

    # (B, H, W, C)
    # -> (B, n_h, patch_size, n_w, patch_size, C)
    patches = image.reshape(B, n_h, patch_size, n_w, patch_size, C)

    # -> (B, n_h, n_w, patch_size, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    # -> (B, N, patch_dim)
    patches = patches.reshape(B, N, patch_dim)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    embeddings = patches @ W_proj  # (B, N, embed_dim)

    return embeddings