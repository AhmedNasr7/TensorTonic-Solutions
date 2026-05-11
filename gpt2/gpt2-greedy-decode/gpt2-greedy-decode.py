import torch



def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.double()
    return torch.tensor(x, dtype=torch.float64)

    
def greedy_decode(input_ids, logits_map, num_steps):
    """
    Returns: list of int (full generated sequence including input_ids)
    """
    # input_ids = tuple(input_ids)

    for i in range(num_steps):
        input_ids_tuple = tuple(input_ids)
        if input_ids_tuple in logits_map:
            logits = logits_map[input_ids_tuple]
            logits = to_tensor(logits)
            token = torch.argmax(logits).item()
            input_ids.append(token)

        else:
            break
            
        
    return input_ids

    

    
    