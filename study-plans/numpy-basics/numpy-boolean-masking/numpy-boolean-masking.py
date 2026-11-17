import numpy as np


def row_summary(data: list, threshold: float) -> np.ndarray:
    values = np.asarray(data)
    mask = values > threshold

    any_row = np.where(mask.any(axis=1, keepdims=True), values, 0)
    all_row = np.where(mask.all(axis=1, keepdims=True), values, 0)

    return np.stack((mask, any_row, all_row)).astype(np.float64)