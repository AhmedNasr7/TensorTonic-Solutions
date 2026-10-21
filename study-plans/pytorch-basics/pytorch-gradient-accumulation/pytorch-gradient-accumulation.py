import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    w = w_init.detach().float().clone().requires_grad_(True)
    avg_grad = torch.zeros_like(w)
    count = 0

    for i, (x, t) in enumerate(micro_batches):
        loss = (w @ x - t) ** 2
        loss.backward()
        count += 1

        if count == accum_steps or i == len(micro_batches) - 1:
            avg_grad = w.grad / count

            with torch.no_grad():
                w -= lr * avg_grad

            w.grad = None
            count = 0

    return w.detach(), avg_grad