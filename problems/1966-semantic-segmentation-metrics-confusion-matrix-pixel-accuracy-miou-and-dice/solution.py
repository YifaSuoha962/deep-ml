import numpy as np

def confusion_matrix(y_true, y_pred, num_classes, ignore_index=255):
    """
    Compute the confusion matrix for semantic segmentation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Mask out ignored pixels
    mask = y_true != ignore_index
    
    # 关键修改：将 int64 改为 np.intp（适配当前平台的指针整数）
    true_valid = y_true[mask].astype(np.intp)
    pred_valid = y_pred[mask].astype(np.intp)
    
    # Map (true, pred) pairs to flat indices and count
    indices = true_valid * num_classes + pred_valid
    cm = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm

def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    num_classes = cm.shape[0]
    total_pixels = cm.sum()
    
    TP = np.diag(cm).astype(np.float64)
    row_sums = cm.sum(axis=1).astype(np.float64)
    col_sums = cm.sum(axis=0).astype(np.float64)
    FN = row_sums - TP
    FP = col_sums - TP
    
    if total_pixels == 0:
        pixel_acc = 0.0
    else:
        pixel_acc = float(TP.sum() / total_pixels)
    
    valid_gt = (TP + FN) > 0
    if valid_gt.any():
        mean_acc = float(np.mean(TP[valid_gt] / (TP[valid_gt] + FN[valid_gt])))
    else:
        mean_acc = 0.0
    
    iou_list = []
    dice_list = []
    for c in range(num_classes):
        denom_iou = TP[c] + FP[c] + FN[c]
        if denom_iou == 0:
            iou_list.append(None)
        else:
            iou_list.append(float(TP[c] / denom_iou))
        
        denom_dice = 2 * TP[c] + FP[c] + FN[c]
        if denom_dice == 0:
            dice_list.append(None)
        else:
            dice_list.append(float(2 * TP[c] / denom_dice))
    
    iou_valid = [v for v in iou_list if v is not None]
    miou = float(np.mean(iou_valid)) if iou_valid else 0.0
    
    dice_valid = [v for v in dice_list if v is not None]
    mean_dice = float(np.mean(dice_valid)) if dice_valid else 0.0
    
    total_gt = TP.sum() + FN.sum()
    if total_gt == 0:
        fwiou = 0.0
    else:
        f = (TP + FN) / total_gt
        iou_array = np.array([v if v is not None else 0.0 for v in iou_list], dtype=np.float64)
        fwiou = float(np.sum(f * iou_array))
    
    return {
        "pixel_acc": pixel_acc,
        "mean_acc": mean_acc,
        "iou": iou_list,
        "dice": dice_list,
        "miou": miou,
        "mean_dice": mean_dice,
        "fwiou": fwiou,
    }

def dataset_metrics(pairs, num_classes, ignore_index=255):
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in pairs:
        total_cm += confusion_matrix(y_true, y_pred, num_classes, ignore_index)
    return metrics_from_confusion(total_cm)