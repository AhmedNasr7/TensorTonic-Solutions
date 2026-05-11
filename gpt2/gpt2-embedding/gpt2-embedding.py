import torch
import torch.nn as nn

def gpt2_embedding(token_ids, token_embed_weight, position_embed_weight):
    """
    Returns: torch.Tensor of shape (seq_len, d_model)
    """

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(data=token_ids, dtype=torch.long)

    if not isinstance(token_embed_weight, torch.Tensor):
        token_embed_weight = torch.tensor(data=token_embed_weight, dtype=torch.float32)

    if not isinstance(position_embed_weight, torch.Tensor):
        position_embed_weight = torch.tensor(data=position_embed_weight, dtype=torch.float32)

    max_indices = len(token_ids)
    indices = torch.arange(max_indices)
    
    outputs = token_embed_weight[token_ids] + position_embed_weight[indices]

    return outputs
