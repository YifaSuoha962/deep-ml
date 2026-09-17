import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Add intercept column
    X_design = np.column_stack([
        np.ones(len(X)),
        X
    ])

    # Solve min ||X_design beta - y||^2
    beta, _, _, _ = np.linalg.lstsq(
        X_design,
        y,
        rcond=None
    )

    return beta


def omitted_variable_bias(
    X: np.ndarray,
    y: np.ndarray,
    omit_idx: int
) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have exactly two columns.")

    if omit_idx not in (0, 1):
        raise ValueError("omit_idx must be 0 or 1.")

    kept_idx = 1 - omit_idx

    # 1. Full regression:
    # y = beta0 + beta1*x1 + beta2*x2
    full_coef = ols(X, y)

    # +1 because coefficient index 0 is the intercept
    full_kept = full_coef[kept_idx + 1]
    beta_omitted = full_coef[omit_idx + 1]

    # 2. Short regression:
    # y = alpha0 + alpha1*x_kept
    X_kept = X[:, [kept_idx]]
    short_coef = ols(X_kept, y)

    short = short_coef[1]

    # 3. Auxiliary regression:
    # x_omitted = gamma0 + delta*x_kept
    omitted_var = X[:, omit_idx]
    aux_coef = ols(X_kept, omitted_var)

    delta = aux_coef[1]

    # Omitted variable bias
    bias = beta_omitted * delta

    return (
        float(full_kept),
        float(short),
        float(bias)
    )