import torch

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)
    
    # Your code here: use autograd to compute gradients
    
    
    for _ in range(iterations):
        pred = X_t @ theta                           # (m, 1)
        loss = (1 / (2 * m)) * torch.sum((pred - y_t) ** 2)   # scalar

        loss.backward()                              # compute gradients
        """
        这只是表示“这次更新操作不记录到计算图中”，并不会把 theta.requires_grad 变成 False。
        """
        with torch.no_grad():
            theta -= alpha * theta.grad              # gradient descent update
            theta.grad.zero_()                       # clear gradients for next iteration
    """
    最后返回的theta是带梯度的，但评测器要求不带梯度
    """
    return theta.detach().flatten()  