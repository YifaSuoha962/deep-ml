import numpy as np


def smote(X_minority, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE.

    Args:
        X_minority:
            2D NumPy array of minority samples,
            shape (n_samples, n_features)
        n_synthetic:
            Number of synthetic samples to generate
        k:
            Number of nearest neighbors to consider

    Returns:
        NumPy array of shape (n_synthetic, n_features)
    """

    X = np.asarray(X_minority, dtype=float)

    if X.ndim != 2:
        raise ValueError("X_minority must be a 2D array.")

    n_samples, n_features = X.shape

    # Boundary case: no synthetic samples requested
    if n_synthetic == 0:
        return np.empty((0, n_features), dtype=float)

    # SMOTE needs at least two original samples
    if n_samples < 2:
        return np.empty((0, n_features), dtype=float)

    k_actual = min(k, n_samples - 1)

    if k_actual <= 0:
        return np.empty((0, n_features), dtype=float)

    synthetics = []

    for _ in range(n_synthetic):

        # 1. Draw base sample index FIRST
        i = np.random.randint(0, n_samples)
        x_i = X[i]

        # Compute Euclidean distances from x_i
        distances = np.sqrt(
            np.sum((X - x_i) ** 2, axis=1)
        )

        # Exclude the sample itself
        distances[i] = np.inf

        # Find k nearest neighbors
        neighbor_indices = np.argsort(distances)[:k_actual]

        # 2. Draw neighbor index SECOND
        neighbor_pos = np.random.randint(0, k_actual)
        x_nn = X[neighbor_indices[neighbor_pos]]

        # 3. Draw interpolation gap THIRD
        gap = np.random.random()

        # SMOTE interpolation
        x_synthetic = x_i + gap * (x_nn - x_i)

        synthetics.append(x_synthetic)

    return np.asarray(synthetics, dtype=float)