import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	
	def decompress_csr(vals_list, col_list, row_ptr, n_cols=None):
		# 将 Compressed Row Sparse (CSR) 表示还原为稠密矩阵。
		n_rows = len(row_ptr) - 1
		# 确定列数
		if n_cols is None:
			n_cols = max(col_list) + 1 if col_list else 0
		# 初始化全零矩阵
		dense_matrix = [[0] * n_cols for _ in range(n_rows)]
		# 逐行填充非零元素
		for i in range(n_rows):
			start = row_ptr[i]
			end = row_ptr[i + 1]
			for k in range(start, end):
				j = col_list[k]
				dense_matrix[i][j] = vals_list[k]
		return dense_matrix
	"""
	n_rows = len(dense_matrix)
	n_cols = len(dense_matrix[0])
	
	vals_list = []
	col_list = []
	row_ptr = [0]
	for i in range(n_rows):
		for j in range(n_cols):
			val = dense_matrix[i][j]
			if val:
				vals_list.append(val)
				col_list.append(j)
		row_ptr.append(len(vals_list))
	return vals_list, col_list, row_ptr


