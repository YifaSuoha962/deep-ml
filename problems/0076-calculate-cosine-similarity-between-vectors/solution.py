import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        v1 (numpy.ndarray): 1D array representing the first vector.
        v2 (numpy.ndarray): 1D array representing the second vector.
    
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # 转换为 numpy 数组
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    # 检查形状是否相同且非空
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape.")
    if v1.size == 0:
        raise ValueError("Vectors cannot be empty.")
    
    # 计算点积
    dot_product = np.dot(v1, v2)
    
    # 计算范数
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    # 检查零幅值
    if v1_norm == 0 or v2_norm == 0:
        raise ValueError("Vectors cannot have zero magnitude.")
    
    return dot_product / (v1_norm * v2_norm)