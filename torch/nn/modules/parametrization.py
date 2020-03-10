from .module import Module


class Parametrization(Module):
    r"""A kind of Module that parametrizes a Parameter in terms of
    a function

    After registering a Parametrization ``P`` on a parameter ``p`` within
    a module ``m`` with :meth:`#Module.register_parametrization`,
    whenever we access ``m.p``, ``P(m.p)`` will be returned

    Parametrizations may be composed by registering several parametriztions
    on the same Parameter.

    Parametrized parameters have an in-built caching system via the
    context manager :class:`torch.nn.cached`.
    """

    def __init__(self):
        super(Parametrization, self).__init__()
        self.caching = False
        self.register_buffer("_cache", None)

    @property
    def cache(self):
        if self.caching:
            return self._cache
        else:
            return self(self.orig)

    @property
    def original(self):
        orig = self._parameters.get("orig")
        if orig is None:
            orig = self._parametrizations["orig"].original
        return orig

    def enable_caching(self):
        self.update_cache()
        self.caching = True

    def disable_caching(self):
        self.caching = False
        self.invalidate_cache()

    def update_cache(self):
        self._cache = self.cache

    def invalidate_cache(self):
        self._cache = None


class cached:
    r"""Context-manager that enables the caching system within
    :class:`torch.nn.Parametrization`.

    This is usful when one uses certain parametrized parameter more than
    once. An example of this is the loop in an RNN model
    """

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.apply(cached._enable_caching)
        return self.model

    def __exit__(self, *args):
        self.model.apply(cached._disable_caching)

    @staticmethod
    def _enable_caching(module):
        # At the moment just the first parametrization is modified,
        # as it is the one that holds the _cache buffer
        for p in module._parametrizations.values():
            p.enable_caching()

    @staticmethod
    def _disable_caching(module):
        for p in module._parametrizations.values():
            p.disable_caching()
