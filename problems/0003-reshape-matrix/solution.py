import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	reshaped_matrix = list()
	num_ori = len(a) * len(a[0])
	num_targ = new_shape[0] * new_shape[1]
	# 判断总量是否相等，相等就变形
	if num_ori == num_targ:
		# 先展平，后切分
		flatten = list()
		for i in range(len(a)):
			flatten.extend(a[i])
		for i in range(new_shape[0]):
			st_idx = i * new_shape[1] 
			reshaped_matrix.append(flatten[st_idx:st_idx+new_shape[1]]) 
	return reshaped_matrix