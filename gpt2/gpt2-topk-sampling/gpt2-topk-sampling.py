import torch

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return x



def apply_temperature(logits, temperature):
    """
    Returns: torch.Tensor of scaled logits
    """
    logits = to_tensor(logits)
    return logits / temperature
    

def top_k_filter(logits, k):
    """
    Returns: torch.Tensor with non-top-k values set to -inf
    """
    if k <= 0:
        return torch.full_like(logits, float("-inf")) 

    k = min(k, logits.size(-1))

    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[..., -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))

def sample_from_logits(logits, random_val):
    """
    Returns: int (sampled token id)
    """
    # YOUR CODE HERE
    probs = torch.nn.functional.softmax(logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    token = torch.searchsorted(cum_probs, random_val).item() # more efficient than torch.where(cum_probs > random_val)[0]

    return token

    
    