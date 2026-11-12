import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    # 每个状态有 n_tilings 个活跃特征，因此按平铺数量缩小步长
    # 切分的是每个权重承担的更新量，让整体学习速度不随 tiling 数量增加而变快。
    adjusted_alpha = base_alpha / n_tilings

    # 使用浮点副本，避免修改调用者的权重列表
    updated_weights = np.array(weights, dtype=float, copy=True)
    indices = np.asarray(active_tiles, dtype=int)

    # 更新前的预测：所有活跃 tile 的权重之和
    prediction_before = float(updated_weights[indices].sum())
    error = target - prediction_before

    # 每个活跃权重增加 alpha * error，非活跃权重保持不变
    updated_weights[indices] += adjusted_alpha * error

    # 使用未舍入的更新结果计算新预测，最后统一保留四位小数
    prediction_after = float(updated_weights[indices].sum())

    return {
        'adjusted_alpha': round(float(adjusted_alpha), 4),
        'prediction_before': round(prediction_before, 4),
        'prediction_after': round(prediction_after, 4),
        'updated_weights': [round(float(w), 4) for w in updated_weights],
    }