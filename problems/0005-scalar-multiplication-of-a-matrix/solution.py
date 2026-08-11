import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	np_mat = np.asarray(matrix)
	scalar_mul = scalar * np_mat
	return scalar_mul
	# pass