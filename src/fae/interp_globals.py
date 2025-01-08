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
        def setterizer(dictionary, layer):
            def setter(value):
                # print("VALUE", {k: v.device for k, v in value.items()})
                dictionary[layer] = value
            return setter
        with self._set_contextvar({
            layer: setterizer(result, layer)
            for layer in layers
        }):
            yield result

    def jax_callback(self, layer_idx, value):
        jax.debug.callback(
            lambda i, a: self._cvar.get(int(i), lambda x: None)(a), layer_idx, value)
        # jax.experimental.io_callback(
        #     lambda i, a: self._cvar.get(int(i), lambda x: None)(a), None, layer_idx, value,
        #     sharding=jax.sharding.SingleDeviceSharding(jax.devices("tpu")[0])
        # )


post_double_stream = InterpHelper("post_double_stream")
post_single_stream = InterpHelper("post_single_stream")
