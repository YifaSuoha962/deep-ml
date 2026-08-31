import numpy as np

def map_latent_to_real(latent_labeled: list, real_labeled: list, latent_query: list) -> np.ndarray:
    """
    Fit an affine map from latent actions to real actions using a small labeled set,
    and apply it to a batch of latent queries.

    Args:
        latent_labeled: (N, d_latent) labeled latent actions.
        real_labeled:   (N, d_real)   corresponding real actions.
        latent_query:   (M, d_latent) latent actions to translate.

    Returns:
        numpy array of shape (M, d_real) with predicted real actions.
    """
    # Convert to numpy arrays
    X = np.asarray(latent_labeled, dtype=float)
    Y = np.asarray(real_labeled, dtype=float)
    Xq = np.asarray(latent_query, dtype=float)

    # Define affine transformation
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])    # [X,1]

    # Solve least squares: X_h @ W = Y
    W, _, _, _ = np.linalg.lstsq(X_h, Y)

    # add bias and predict
    #[X,1]⋅[W, b^{\top}]
    Xq_h = np.hstack([Xq, np.ones((Xq.shape[0], 1))])
    Y_pred = Xq_h @ W
    return Y_pred.tolist()
