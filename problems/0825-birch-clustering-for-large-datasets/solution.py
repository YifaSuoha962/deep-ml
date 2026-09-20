import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.
    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius
    Returns: list of centroids (each a list of floats), sorted lexicographically.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    subclusters = []

    def make_cf(x):
        N = 1
        LS = x.copy()   # linear sum
        SS = x ** 2     # squared sum
        return [N, LS, SS]  # tuple definition: (N,LS,SS)
    
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

    def merge_cf(cf1, cf2):
        N_1, LS_1, SS_1 = cf1
        N_2, LS_2, SS_2 = cf2
        return [N_1 + N_2, LS_1 + LS_2, SS_1 + SS_2]

    for x in X:
        tmp_cf = make_cf(x)
        
        # case 1
        if not subclusters:
            subclusters.append(tmp_cf)
            # jump over the following steps
            continue
        # case 2
        # 2.1 traverse all subclusters and calculate pairwise distance
        distances = []
        for cf in subclusters:
            centroid_cf = centroid(cf)
            dist = np.sqrt(np.sum((centroid_cf - x) ** 2))  # Euclidean distance: sum along the feature dim
            distances.append(dist)
        # np.argmin naturally breaks ties by smaller index
        """
        tip: how to find the index corresponding to the expected value (min /max)
        """
        # 2.2 find the closest subcluster
        nearest_idx = int(np.argmin(distances))
        
        # 3. absorb x into that subcluster
        """
        Note the criteria: If the resulting radius ≤ threshold, keep the absorption,
        otherwise create a new one.
        """
        res_cf = merge_cf(tmp_cf, subclusters[nearest_idx])
        if radius(res_cf) <= threshold:
            subclusters[nearest_idx] = res_cf
        else:
            subclusters.append(tmp_cf)
    
    # 4. Compute final centroids
    centroids = [
        centroid(cf).tolist()
        for cf in subclusters
    ]

    # Lexicographic ascending sort
    centroids.sort()

    return centroids




