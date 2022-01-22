from deep_q.embeddingchars import EmbeddingChars
from deep_q.mlp import MLP
from deep_q.sumchars import SumChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(MLP, "MLP")
register(EmbeddingChars, "EmbeddingChars")
register(SumChars, "SumChars")
