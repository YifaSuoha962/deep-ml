import torch

def calculate_auc(y_true, y_scores) -> float:
    """
    计算 ROC 曲线下面积 (AUC)。
    
    Args:
        y_true: 二值真实标签列表、数组或张量 (0 或 1)
        y_scores: 预测概率或置信度分数
        
    Returns:
        AUC 值 (float)
    """
    # 转换为张量
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_scores = torch.as_tensor(y_scores, dtype=torch.float32)
    
    # 正负样本数
    P = y_true.sum().item()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return 0.0  # 边界情况：全为同一类
    
    # 按分数降序排序
    sorted_indices = torch.argsort(y_scores, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # 计算每个阈值下的 TPR 和 FPR
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    # 我们只关心被预测为正的样本中有多少是真正的正例（TP）和负例（FP）
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
    
    # 梯形积分计算 AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    
    return auc