import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    # boundary case: 如果两个列表都为空，返回 0.0
    if len(predictions) == 0 and len(references) == 0:
        return 0.0
    
    # 检查长度是否一致，不一致则抛出异常（也可按较短长度处理）
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    
    def normalize(text: str) -> str:
        # 1. 转换为小写
        text = text.lower()
        # 2. 去除所有标点字符
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 3. 压缩多个空白字符为单个空格，并去除首尾空白
        text = ' '.join(text.split())
        return text

    # 对每个预测和参考进行规范化，并逐对比较
    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize(pred) == normalize(ref):
            matches += 1
    
    # 返回匹配比例
    return matches / len(predictions)
