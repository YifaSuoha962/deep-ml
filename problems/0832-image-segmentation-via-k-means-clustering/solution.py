import numpy as np

def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.

    Args:
        image: nested list / array of shape (H, W, 3) with pixel values.
        k: number of clusters.
        max_iters: maximum number of K-Means iterations.

    Returns:
        Segmented image as a nested list of shape (H, W, 3).
    """
    pixels = np.asarray(image, dtype=float)

    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    
    # 按行优先顺序取前 k 个像素作为初始质心
    height, width, _ = pixels.shape
    flat_pixels = pixels.reshape(-1, 3)     # -> [H * W, 3] 
    if not 1 <= k <= len(flat_pixels):
        raise ValueError("k must be between 1 and the number of pixels")
    centroids = flat_pixels[:k].copy()

    for _ in range(max_iters):
        # 计算每个像素到每个质心的平方欧氏距离  (pairwise-distance)。
        # np.argmin 遇到相同最小值时会返回最先出现的索引，因此平局时选最低索引。
        distances = np.sum(
            (flat_pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        assignments = np.argmin(distances, axis=1)   # [H*W, 1] -- elem is index

        updated_centroids = centroids.copy()
        for cluster_idx in range(k):
            cluster_pixels = flat_pixels[assignments == cluster_idx]
            if len(cluster_pixels) > 0:
                updated_centroids[cluster_idx] = cluster_pixels.mean(axis=0)
            
        # 空簇保留之前的质心
        unchanged = np.array_equal(updated_centroids, centroids)
        centroids = updated_centroids

        if unchanged:
            break

    # 用最终质心重新分配像素，再生成分割图
    final_distances = np.sum(
        (flat_pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    final_assignments = np.argmin(final_distances, axis=1)
    segmented = centroids[final_assignments].reshape(height, width, 3)

    return np.round(segmented, 4).tolist()