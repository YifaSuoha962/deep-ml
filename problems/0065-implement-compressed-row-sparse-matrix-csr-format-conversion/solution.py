def compressed_row_sparse_matrix(dense_matrix):
    """
    将稠密矩阵转换为 Compressed Row Sparse (CSR) 表示。
    :param dense_matrix: 二维列表表示的稠密矩阵
    :return: (values, column_indices, row_pointer) 三个列表
    """
    vals_list = []
    cols_list = []
    row_ptr = [0]  # 行指针，初始为 0

    for i in range(len(dense_matrix)):
        count = 0  # 当前行的非零元素个数
        for j in range(len(dense_matrix[i])):
            val = dense_matrix[i][j]
            if val:  # 非零元素（0 视为假）
                vals_list.append(val)
                cols_list.append(j)
                count += 1
        # 累加当前行的非零元素个数到行指针
        row_ptr.append(row_ptr[-1] + count)

    return vals_list, cols_list, row_ptr