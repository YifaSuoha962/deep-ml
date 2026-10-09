def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	n_rows = len(dense_matrix)
	n_cols = len(dense_matrix[0])

	values = []
	row_indices = []
	col_ptr = [0]

	cnt = 0
	for j in range(n_cols):
		for i in range(n_rows):
			val = dense_matrix[i][j]
			if val:
				values.append(val)
				row_indices.append(i)
				cnt += 1
		col_ptr.append(cnt)
	return values, row_indices, col_ptr