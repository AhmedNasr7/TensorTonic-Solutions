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

import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """

    B, N, D = patches.shape
    if cls_token is None:
        cls_token = np.random.randn(1, 1, embed_dim) * 0.02

    cls_tokens = np.tile(cls_token, (B, 1, 1))

    patches = np.concatenate([cls_tokens, patches], axis=1)

    return patches




def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    """
    Add position embeddings to patch embeddings.
    pos_embed: position embedding of shape (1, N, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    if pos_embed is None:
        np.random.randn(1, num_patches, embed_dim) * 0.02

    return patches + pos_embed

    
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

    

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2 + 1
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        patch_dim = patch_size * patch_size * 3
        hidden_dim = int(embed_dim * mlp_ratio)

        # Patch projection: flattened patch -> embedding
        self.W_patch = (
            W_patch
            if W_patch is not None
            else np.random.randn(patch_dim, embed_dim) * 0.02
        )

        # CLS token: one learnable token shared across batch
        self.cls_token = (
            cls_token
            if cls_token is not None
            else np.random.randn(1, 1, embed_dim) * 0.02
        )

        # Positional embedding: num_patches + CLS token
        self.pos_embed = (
            pos_embed
            if pos_embed is not None
            else np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        # Encoder block weights
        if encoder_weights is not None:
            self.encoder_weights = encoder_weights
        else:
            self.encoder_weights = []

            for _ in range(depth):
                block_weights = {
                    "Wq": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wk": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wv": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wo": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "W1": np.random.randn(embed_dim, hidden_dim) * 0.02,
                    "W2": np.random.randn(hidden_dim, embed_dim) * 0.02,
                }

                self.encoder_weights.append(block_weights)

        # Classification head: CLS embedding -> class logits
        self.W_head = (
            W_head
            if W_head is not None
            else np.random.randn(embed_dim, num_classes) * 0.02
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """

        x_embed_patches = patch_embed(x, patch_size=self.patch_size, embed_dim=self.embed_dim ,W_proj=self.W_patch)

        patches = prepend_class_token(patches=x_embed_patches, embed_dim=self.embed_dim, cls_token=self.cls_token)

        patches = add_position_embedding(patches=patches, num_patches=self.num_patches, embed_dim=self.embed_dim, pos_embed=self.pos_embed)

        for block_weights in self.encoder_weights:
            patches = vit_encoder_block(
                patches,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                Wq=block_weights["Wq"],
                Wk=block_weights["Wk"],
                Wv=block_weights["Wv"],
                Wo=block_weights["Wo"],
                W1=block_weights["W1"],
                W2=block_weights["W2"],
            )


        outputs = classification_head(encoder_output=patches, num_classes=self.num_classes, W_head=self.W_head)


        return outputs