def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    # 将输入点从元组转换为列表，以便后续修改和计算（列表是可变的）
    points_list = [list(p) for p in points]
    # 获取点的维度（第一个点的长度），如果点集为空则维度为0
    dim = len(points[0]) if points else 0
    # 将初始质心从元组转换为列表，便于更新
    centroids = [list(c) for c in initial_centroids]

    # 进行最多 max_iterations 次迭代
    for _ in range(max_iterations):
        # 为每个簇准备一个空列表，用于存储分配给该簇的点
        clusters = [[] for _ in range(k)]
        
        # 遍历每个点，将其分配给最近的质心
        for pt in points_list:
            # 初始化最小距离为正无穷
            min_dist = float('inf')
            best_idx = 0  # 记录最佳质心的索引
            
            # 遍历所有质心，找到距离当前点最近的质心
            for idx, c in enumerate(centroids):
                # 计算平方欧几里得距离（避免开平方，节省计算）
                dist_sq = sum((pt[i] - c[i]) ** 2 for i in range(dim))
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best_idx = idx
            # 将当前点添加到最近质心对应的簇中
            clusters[best_idx].append(pt)

        # 更新质心：每个簇的新质心是该簇所有点的均值
        new_centroids = []
        for idx in range(k):
            if clusters[idx]:  # 如果该簇非空
                # 计算每个维度的平均值
                mean = [sum(pt[i] for pt in clusters[idx]) / len(clusters[idx]) for i in range(dim)]
                new_centroids.append(mean)
            else:
                # 如果簇为空，则保留旧质心（避免质心消失）
                new_centroids.append(centroids[idx])

        # 检查收敛：如果所有质心的变化都小于 1e-6，则停止迭代
        converged = True
        for i in range(k):
            for j in range(dim):
                if abs(new_centroids[i][j] - centroids[i][j]) > 1e-6:
                    converged = False
                    break
            if not converged:
                break
        
        # 更新质心为新的质心
        centroids = new_centroids
        
        # 如果已收敛，提前终止迭代
        if converged:
            break

    # 将最终质心四舍五入到小数点后4位，并转换回元组形式
    final_centroids = [tuple(round(c, 4) for c in cent) for cent in centroids]
    return final_centroids