from dqn.embeddingchars import EmbeddingChars
from dqn.mlp import MLP
from dqn.sumchars import SumChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(MLP, "MLP")
register(EmbeddingChars, "EmbeddingChars")
register(SumChars, "SumChars")
