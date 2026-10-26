import numpy as np
from scipy.special import log_softmax

def label_smoothing(logits: np.ndarray, y: int, eps: float) -> tuple[np.ndarray, float]:
    C = len(logits)

    # Start by giving every class eps/C
    q = np.full(C, eps / C)

    # Add remaining mass to the true class
    q[y] += (1 - eps)

    # Stable log-softmax
    log_probs = log_softmax(logits)

    # Cross-entropy
    loss = -np.sum(q * log_probs)

    return q, float(loss)
