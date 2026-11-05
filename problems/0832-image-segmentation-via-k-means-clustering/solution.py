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
    # centroids 根据 final_assignments中的索引 [0, k-1]，收集每个 pixel 对应的 centriod feature
    # 最后还是形成了 H*W 个特征
    segmented = centroids[final_assignments].reshape(height, width, 3)

    return np.round(segmented, 4).tolist()


# import torch

# def segment_image(image, k: int, max_iters: int = 100):
#     """
#     使用 K-Means 按 RGB 颜色对图像进行分割。

#     Args:
#         image: 形状为 (H, W, 3) 的嵌套列表、NumPy 数组或 PyTorch 张量。
#         k: 簇的数量。
#         max_iters: K-Means 的最大迭代次数。

#     Returns:
#         形状为 (H, W, 3) 的嵌套列表。
#     """
#     # 转成浮点张量。若 image 已是张量，会保留它所在的设备。
#     pixels = torch.as_tensor(image, dtype=torch.float64)

#     # 确认输入是 RGB 图像。
#     if pixels.ndim != 3 or pixels.shape[2] != 3:
#         raise ValueError("image must have shape (H, W, 3)")

#     # 记录图像尺寸，并将像素展平成 (H*W, 3)。
#     height, width, _ = pixels.shape
#     flat_pixels = pixels.reshape(-1, 3)

#     # 确认簇数量有效。
#     if not 1 <= k <= flat_pixels.shape[0]:
#         raise ValueError("k must be between 1 and the number of pixels")

#     # 按行优先顺序取前 k 个像素作为初始质心。
#     centroids = flat_pixels[:k].clone()

#     for _ in range(max_iters):
#         # 扩展维度后广播相减：
#         # flat_pixels[:, None, :] 的形状为 (H*W, 1, 3)
#         # centroids[None, :, :] 的形状为 (1, k, 3)
#         # 差值的形状为 (H*W, k, 3)。
#         distances = torch.sum(
#             (flat_pixels[:, None, :] - centroids[None, :, :]) ** 2,
#             dim=2,
#         )

#         # 在每个像素对应的 k 个距离中找最小值索引。
#         # assignments 的形状是 (H*W,)，不是 (H*W, k)。
#         # 若最小值并列，argmin 返回第一个最小值的索引。
#         assignments = torch.argmin(distances, dim=1)

#         # 先复制旧质心，这样没有像素的空簇会保留原质心。
#         updated_centroids = centroids.clone()

#         for cluster_idx in range(k):
#             # 布尔掩码选出属于当前簇的像素。
#             cluster_pixels = flat_pixels[assignments == cluster_idx]

#             # 非空簇的新质心等于簇内像素的均值。
#             if cluster_pixels.shape[0] > 0:
#                 updated_centroids[cluster_idx] = cluster_pixels.mean(dim=0)

#         # 所有质心都未变化时，提前结束迭代。
#         unchanged = torch.equal(updated_centroids, centroids)
#         centroids = updated_centroids

#         if unchanged:
#             break

#     # 用最终质心重新分配每个像素。
#     final_distances = torch.sum(
#         (flat_pixels[:, None, :] - centroids[None, :, :]) ** 2,
#         dim=2,
#     )
#     final_assignments = torch.argmin(final_distances, dim=1)

#     # 根据每个像素的簇编号取出对应质心，得到 (H*W, 3)，再还原图像形状。
#     # 扩展为 (H*W, 3)，每行都包含该像素的质心索引。
#     gather_indices = final_assignments.unsqueeze(1).expand(-1, 3)

#     # 从 centroids (k, 3) 中按行收集，结果形状为 (H*W, 3)。
#     segmented = torch.gather(centroids, dim=0, index=gather_indices)

#     # 保留 4 位小数并转换成嵌套 Python 列表。
#     return torch.round(segmented, decimals=4).tolist()