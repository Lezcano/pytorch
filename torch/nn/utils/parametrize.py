r"""
Parametrization methods
"""
from abc import abstractmethod
import numbers
import torch
import torch.nn as nn
from ..modules.module import SubParametrizationList, ParametrizationList
from torch._jit_internal import _copy_to_script_wrapper

# For Python 2 and 3 support
try:
    from abc import ABC
    from collections.abc import Iterable
except ImportError:
    from abc import ABCMeta
    ABC = ABCMeta('ABC', (), {})
    from collections import Iterable


# Short example
#x = nn.Linear(3, 4)
#orth = Orthogonal(*args, **kwargs)
## Creates Base and whatever mad shit it needs
## and saves them in orth
#orth = x.register_parametrization(orth, "weight")
#Orthogonal.torus_init_(x.weight, orth.parameters())


class BaseParametrization(nn.Module):
    LIST_TYPE = SubParametrizationList

    def init_parameters(self, original_tensor):
        pass


# TODO Maybe make it (if possible) so that we can apply it to a model as
# @cached
# class Model(nn.Module):
class cached:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.apply(cached._set_cache_in)
        return self.model

    def __exit__(self, *args):
        self.model.apply(cached._set_cache_out)

    @staticmethod
    def _set_cache_in(module):
        if isinstance(module, ParametrizationList):
            module.update_cache()
            module.caching = True

    @staticmethod
    def _set_cache_out(module):
        if isinstance(module, ParametrizationList):
            module.caching = False
            module.invalidate_cache()
