import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks, x, use_checkpoint=False):
    # TODO: apply each block in sequence to x
    # TODO: when use_checkpoint is True, run each block through
    #       checkpoint(...) with use_reentrant explicitly set to False
    for blk in blocks:
        if use_checkpoint:
            x = checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
    return x
