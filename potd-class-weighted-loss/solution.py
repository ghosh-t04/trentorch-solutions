import numpy as np

def class_weighted_loss(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:

    # Step 1: Make logits numerically stable
    logits = logits - np.max(logits, axis=1, keepdims=True)

    # Step 2: Calculate log-softmax directly
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))

    # Step 3: Extract log probability of correct class
    correct_log_probs = log_probs[np.arange(len(labels)), labels]

    # Step 4: Calculate negative log likelihood
    losses = -correct_log_probs

    # Step 5: Apply class weights
    weighted_losses = losses * weights[labels]

    # Step 6: Calculate mean weighted loss
    final_loss = np.sum(weighted_losses) / len(labels)

    return float(final_loss)
