"""Interface for poison recipes."""
from .forgemaster_targeted import ForgemasterTargeted

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'targeted':
        return ForgemasterTargeted(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']
