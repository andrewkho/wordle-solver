from networks.embeddingchars import EmbeddingChars
from networks.mlp import MLP
from networks.sumchars import SumChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(MLP, "MLP")
register(EmbeddingChars, "EmbeddingChars")
register(SumChars, "SumChars")
