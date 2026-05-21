import numpy as np


def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    return - 0.5 * np.mean(np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=-1), axis=0)



def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    recon = np.mean(np.sum((x - x_recon) ** 2, axis=-1), axis=0)
    kl = kl_divergence(mu, log_var)
    total = recon + kl 
    return {"total": total,
            "recon": recon, 
            "kl": kl
           }
    
