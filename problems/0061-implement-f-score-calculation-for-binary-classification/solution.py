import numpy as np

def f_score(y_true, y_pred, beta):
    # 向量化计算 TP, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # 加入极小值 eps 防止分母为 0 导致报错
    eps = 1e-10
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    
    beta_sq = beta ** 2
    F_b = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)
    
    return round(float(F_b), 3)