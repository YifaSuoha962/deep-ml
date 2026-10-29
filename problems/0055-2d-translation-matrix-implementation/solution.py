import torch

def translate_object(points, tx, ty) -> torch.Tensor:
    """
    Apply a 2D translation to a set of points.

    Args:
        points: list of [x, y] coordinates or a tensor of shape (N, 2)
        tx: translation distance in the x direction
        ty: translation distance in the y direction

    Returns:
        Tensor of translated points with shape (N, 2)
    """
    points = torch.as_tensor(points)

    # Ensure integer inputs can still represent fractional translations.
    if not points.is_floating_point():
        points = points.to(torch.float32)

    # Handle an empty list as a tensor with shape (0, 2).
    if points.numel() == 0:
        points = points.reshape(0, 2)

    translation = torch.tensor(
        [tx, ty],
        dtype=points.dtype,
        device=points.device,
    )

    return points + translation