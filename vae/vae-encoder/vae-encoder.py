import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns: dict with 'mu' and 'log_var' as np.ndarrays of shape (batch, latent_dim)
    """
    # Your implementation here
    N, D = x.shape
    mu = np.matmul(x, W_mu) + b_mu
    log_var = np.matmul(x, W_logvar) + b_logvar

    return {"mu": mu,
            "log_var": log_var}
    
    
    
