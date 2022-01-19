from deep_q.q_networks.embeddingchars import EmbeddingChars
from deep_q.q_networks.mlp import MLP
from deep_q.q_networks.sumchars import SumChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(MLP, "MLP")
register(EmbeddingChars, "EmbeddingChars")
register(SumChars, "SumChars")
