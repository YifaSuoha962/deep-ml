import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	num_pred = y_pred.shape[0]
	# y_true = 1
	TP = 0
	FN = 0
	# y_true = 0
	FP = 0
	TN = 0
	for i in range(num_pred):
		if y_true[i] == 1 and y_pred[i] == 1:
			TP += 1
		elif y_true[i] == 1 and y_pred[i] == 0:
			FN += 1
		elif y_true[i] == 0 and y_pred[i] == 1:
			FP += 1
		else:
			TN += 1
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	F_b = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) 
	return round(F_b, 3)
