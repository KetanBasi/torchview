from __future__ import annotations

import sys

from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar, MutableSet, Any, Generator

from torch import nn
from torch.nn.parameter import Parameter

try:
    import transformers as tr
except ImportError:
    pass

T = TypeVar('T')


class OrderedSet(MutableSet[T]):
    '''Ordered set using mutableset. This is necessary for having reproducible
    iteration order. This property is useful for getting reproducible
    text representation of graphviz graphs, to be used in tests. This is
    because in algorith to produce graph many set objects are iterated.'''
    def __init__(self, iterable: Any | None = None) -> None:
        self.map: dict[T, None] = {}
        if iterable is not None:
            self |= iterable

    def __len__(self) -> int:
        return len(self.map)

    def __contains__(self, value: object) -> bool:
        return value in self.map

    def add(self, value: T) -> None:
        if value not in self.map:
            self.map[value] = None

    def remove(self, value: T) -> None:
        if value in self.map:
            _ = self.map.pop(value)

    def discard(self, value: T) -> None:
        if value in self.map:
            _ = self.map.pop(value)

    def __iter__(self) -> Generator[T, Any, Any]:
        for cur in self.map:
            yield cur

    def __repr__(self) -> str:
        if not self:
            return f'{self.__class__.__name__}'
        return f'{self.__class__.__name__}({list(self)})'


def is_generator_empty(parameters: Iterable[Parameter]) -> bool:
    try:
        _ = next(iter(parameters))
        return False
    except StopIteration:
        return True


def updated_dict(
    arg_dict: dict[str, Any], update_key: str, update_value: Any
) -> dict[str, Any]:
    return {
        keyword: value if keyword != update_key else update_value
        for keyword, value in arg_dict.items()
    }


def assert_input_type(
    func_name: str, valid_input_types: tuple[type, ...], in_var: Any
) -> None:

    assert isinstance(in_var, valid_input_types), (
        f'For an unknown reason, {func_name} function was '
        f'given input with wrong type. The input is of type: '
        f'{type(in_var)}. But, it should be {valid_input_types}'
    )


def get_torch_layer_types() -> dict:
    '''Get mapping of layer types to their respective modules.'''
    _layer_types = dict()
    # _layer_modules = set(
    #     t for t in nn.modules.__dir__()
    #     if type(getattr(nn.modules, t)).__name__ == "module"
    # ) - {"_functions", "loss", "module", "utils"}
    _layer_modules = {
        "activation", "adaptive", "batchnorm",
        "channelshuffle", "container", "conv", "distance",
        "dropout", "flatten", "fold", "instancenorm", "lazy",
        "linear", "normalization", "padding", "pixelshuffle",
        "pooling", "rnn", "sparse", "transformer", "upsampling"
    }

    for _name in _layer_modules:
        _module = getattr(nn.modules, _name)
        if not hasattr(_module, '__all__'):
            continue

        for module in _module.__all__:
            try:
                _layer_types[getattr(nn, module).__name__] = _name
            except AttributeError:
                continue

    return _layer_types


def get_hf_layer_types() -> dict:
    '''Get mapping of layer types to their respective modules.'''
    _layer_types = dict()

    if 'transformers' not in sys.modules:
        return _layer_types

    # TODO: Add TF support (e.g. from `hf.activations_tf.ACT2FN`)
    _name = "activation"
    modules = tr.activations.ACT2CLS

    for _, module in modules.items():
        module = module[0] if isinstance(module, tuple) else module
        _layer_types[module.__name__] = _name

    return _layer_types


def get_layer_types() -> dict:
    '''Get mapping of layer types to their respective modules.'''
    _layer_types = get_torch_layer_types()
    _layer_types.update(get_hf_layer_types())
    return _layer_types
