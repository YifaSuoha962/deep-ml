import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    n_samples, n_features = X.shape
    residuals = y - current_predictions

    best_mse = float("inf")
    best_feature = None
    best_threshold = None
    best_left_value = None
    best_right_value = None

    for feature in range(n_features):
        values = X[:, feature]
        # 去重的目的：重复值不会带来新的分组；若不去重，可能会生成重复或无效的候选阈值。
        unique_vals = np.unique(values)
        # 至少需要两个不同取值才能划分
        if len(unique_vals) < 2:
            continue
        # 候选阈值：相邻唯一值的中点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for threshold in thresholds:
            # 根据阈值将数组值域划到左右区间
            left_mask = values <= threshold
            right_mask = ~left_mask
            # 左右两边都必须至少有一个样本
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            left_value = residuals[left_mask].mean()    # c_L
            right_value = residuals[right_mask].mean()  # c_R

            stump_preds = np.empty(n_samples, dtype=float)
            stump_preds[left_mask] = left_value
            stump_preds[right_mask] = right_value

            mse = np.mean((residuals - stump_preds) ** 2)

            # 严格小于才更新，平局保留先遇到的划分
            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_threshold = threshold
                best_left_value = left_value
                best_right_value = right_value

    # 没有有效划分时，预测所有残差的均值
    if best_feature is None:
        stump_preds = np.full(n_samples, residuals.mean())
    else:
        left_mask = X[:, best_feature] <= best_threshold
        stump_preds = np.empty(n_samples, dtype=float)
        stump_preds[left_mask] = best_left_value
        stump_preds[~left_mask] = best_right_value

    updated_predictions = current_predictions + learning_rate * stump_preds

    return [round(float(v), 4) for v in updated_predictions]
        