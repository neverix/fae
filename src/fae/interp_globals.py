# ...
# this is not a place of honor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import jax.experimental
import jax


class InterpHelper(object):
    def __init__(self, name):
        self._cvar = ContextVar(name, default={})

    @contextmanager
    def _set_contextvar(self, value):
        token = self._cvar.set(self._cvar.get() | value)
        try:
            yield
        finally:
            self._cvar.reset(token)

    @contextmanager
    def capture(self, *layers):
        result = {}
        with self._set_contextvar({
            layer: partial(result.__setitem__, layer)
            for layer in layers
        }):
            yield result

    def jax_callback(self, layer_idx, value, depth):
        jax.lax.switch(layer_idx, [
            (lambda a:
                jax.experimental.io_callback(
                    self._cvar.get().get(i, lambda x: None), None, a))
            for i in range(depth)
        ], value)


post_double_stream = InterpHelper('post_double_stream')
