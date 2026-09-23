import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks, x, use_checkpoint=False):
    """
    Apply each module in `blocks` sequentially to `x`.

    Args:
        blocks: iterable of nn.Module objects
        x: input tensor
        use_checkpoint: whether to use activation checkpointing

    Returns:
        Output tensor after applying all blocks.
    """

    for block in blocks:
        if use_checkpoint:
            x = checkpoint(
                block,
                x,
                use_reentrant=False
            )
        else:
            x = block(x)

    return x