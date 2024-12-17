# ...
# this is not a place of honor
from contextlib import contextmanager
from functools import partial
import jax.experimental
from copy import copy
import jax


class InterpHelper(object):
    def __init__(self, name):
        self._cvar = {}

    @contextmanager
    def _set_contextvar(self, value):
        old = copy(self._cvar)
        self._cvar = old | value
        try:
            yield
        finally:
            self._cvar = old

    @contextmanager
    def capture(self, *layers):
        result = {}
        with self._set_contextvar({
            layer: (lambda x: partial(result.__setitem__, layer)(x))
            for layer in layers
        }):
            yield result

    def jax_callback(self, layer_idx, value):
        jax.debug.callback(
            lambda i, a: self._cvar.get(int(i), lambda x: None)(a), layer_idx, value)
        # jax.experimental.io_callback(
        #     lambda i, a: self._cvar.get(int(i), lambda x: None)(a), None, layer_idx, value)


post_double_stream = InterpHelper('post_double_stream')
