import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.

    X: array-like of shape (n_samples, n_features)

    threshold: float, max allowed subcluster radius

    Returns:
        list of centroids (each a list of floats),
        sorted lexicographically.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    subclusters = []

    def make_cf(x):
        N = 1
        LS = x.copy()
        SS = x ** 2
        return [N, LS, SS]

    def centroid(cf):
        N, LS, SS = cf
        return LS / N

    def radius(cf):
        N, LS, SS = cf

        mean = LS / N
        variance = SS / N - mean ** 2

        # Clamp tiny negative values caused by floating-point error
        variance = np.maximum(variance, 0.0)

        return np.sqrt(np.sum(variance))

    for x in X:

        # 1. First point creates the first subcluster
        if not subclusters:
            subclusters.append(make_cf(x))
            continue

        # 2. Find closest centroid
        distances = []

        for cf in subclusters:
            c = centroid(cf)
            dist = np.sqrt(np.sum((x - c) ** 2))
            distances.append(dist)

        # np.argmin naturally breaks ties by smaller index
        nearest_idx = int(np.argmin(distances))

        # 3. Tentatively absorb x
        N, LS, SS = subclusters[nearest_idx]

        new_cf = [
            N + 1,
            LS + x,
            SS + x ** 2
        ]

        # Keep absorption only if radius <= threshold
        if radius(new_cf) <= threshold:
            subclusters[nearest_idx] = new_cf
        else:
            subclusters.append(make_cf(x))

    # 4. Compute final centroids
    centroids = [
        centroid(cf).tolist()
        for cf in subclusters
    ]

    # Lexicographic ascending sort
    centroids.sort()

    return centroids