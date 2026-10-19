import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    abs_diff = np.abs(y_true_arr - y_pred_arr)

    # 小误差用平方项，大误差用线性项；<= delta 对应平方项
	
    loss = np.where(
        abs_diff <= delta,
        0.5 * abs_diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    )

    return float(np.mean(loss))