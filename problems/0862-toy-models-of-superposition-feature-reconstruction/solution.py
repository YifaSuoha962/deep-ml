import numpy as np

def relu(x):
    return np.maximum(0, x)


def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.

    Args:
        W: array of shape (n_hidden, n_features)
        b: array of shape (n_features,)
        X: array of shape (batch_size, n_features)

    Returns:
        list of lists of shape (batch_size, n_features) with reconstructed features
    """
    W = np.asarray(W)
    b = np.asarray(b)
    X = np.asarray(X)

    output = relu(X @ W.T @ W + b)
    return output