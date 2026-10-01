import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Poisson deviance:

        D = 2 * sum[
            y_i * log(y_i / mu_i) - (y_i - mu_i)
        ]

    using the convention:
        0 * log(0) = 0
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # y and mu must have the same shape
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")

    # Poisson fitted means must be positive
    if np.any(mu <= 0):
        raise ValueError("mu must contain only positive values")

    # --------------------------------------------
    # Boundary case:
    # if y_i = 0,
    # y_i * log(y_i / mu_i) is defined as 0
    #
    # Do not directly compute log(0), masked -- acting as adding 0.
    # --------------------------------------------
    log_term = np.zeros_like(y, dtype=float)

    mask = y > 0
    log_term[mask] = (
        y[mask] * np.log(y[mask] / mu[mask])
    )

    deviance = 2.0 * np.sum(
        log_term - (y - mu)
    )

    return float(deviance)


def dispersion_ratio(
    y: np.ndarray,
    mu: np.ndarray,
    n_params: int
) -> float:
    """
    Pearson chi-square divided by degrees of freedom:

        phi =
            sum((y_i - mu_i)^2 / mu_i)
            / (n - n_params)
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # y and mu must have the same shape
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")

    # Avoid division by zero
    if np.any(mu <= 0):
        raise ValueError("mu must contain only positive values")

    n = y.size
    dof = n - n_params

    if dof <= 0:
        raise ValueError(
            "n_params must be smaller than the number of observations"
        )

    pearson_chi_square = np.sum(
        (y - mu) ** 2 / mu
    )

    return float(
        pearson_chi_square / dof
    )