import numpy as np

def fit_sigmoid_scaling(nll_list, acc_list, predict_nll):
    """
    Fit acc = 1 / (1 + exp(a*nll + b)) via least squares in logit space and
    predict accuracy at predict_nll.

    Returns:
        [a, b, predicted_acc] as a list of floats.
    """
    # 转换为 NumPy 数组（双精度）
    nll = np.array(nll_list, dtype=np.float64).reshape(-1, 1)
    acc = np.array(acc_list, dtype=np.float64)

    # 计算 logit：z = log((1 - acc) / acc)
    z = np.log((1 - acc) / acc)

    # 构造设计矩阵 X = [nll, 1]
    X = np.hstack([nll, np.ones_like(nll)])  # shape (N, 2)

    # 最小二乘求解 X @ theta = z
    theta, _, _, _ = np.linalg.lstsq(X, z.reshape(-1, 1), rcond=None)

    a = theta[0, 0]
    b = theta[1, 0]

    # 预测给定 nll 的准确率（矩阵乘法形式）
    X_pred = np.array([[predict_nll, 1.0]], dtype=np.float64)
    logit = X_pred @ theta  # (1, 1)
    pred = 1.0 / (1.0 + np.exp(logit))
    pred_val = pred[0, 0]

    return [float(a), float(b), float(pred_val)]