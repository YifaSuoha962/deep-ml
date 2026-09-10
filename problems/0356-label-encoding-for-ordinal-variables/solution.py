import torch

def label_encode_ordinal(values: list, order: list) -> torch.Tensor:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        torch.Tensor of integers (dtype=torch.long) representing the encoded values,
        with -1 for any value not found in order
    """
    mapping = {cat: idx for idx, cat in enumerate(order)}
    labels_list = [mapping.get(val, -1) for val in values]
    return torch.tensor(labels_list, dtype=torch.long)