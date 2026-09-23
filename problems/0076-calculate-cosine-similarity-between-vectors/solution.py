import torch

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity between two vectors using PyTorch.
    
    Args:
        v1: 1D array-like representing the first vector.
        v2: 1D array-like representing the second vector.
    
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # 转换为 torch 张量（float32）
    v1 = torch.as_tensor(v1, dtype=torch.float32)
    v2 = torch.as_tensor(v2, dtype=torch.float32)
    
    # 计算点积
    dot_product = torch.dot(v1, v2)
    
    # 计算 L2 范数
    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)
    
    similarity = dot_product / (v1_norm * v2_norm)
    return similarity.item()