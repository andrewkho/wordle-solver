from a2c.sumchars import SumChars
from a2c.embeddingchars import EmbeddingChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(SumChars, "SumChars")
register(EmbeddingChars, "EmbeddingChars")
