# ...
# this is not a place of honor
from contextlib import contextmanager
from functools import partial
import jax.experimental
from copy import copy
from oryx.core.interpreters.harvest import call_and_reap, sow
import jax


REAP_MAX_N = 128


def named_sow(x, key):
    return sow(x, tag="interp", name=key)


def cond_sower(index, sower, key="", max_n=REAP_MAX_N):
    return lambda x: jax.lax.switch(
        index,
        [partial(sower, key=key + f".{i}")
         for i in range(max_n)], x
    )


class Reaper(object):
    def __init__(self, base_key, max_n: int = REAP_MAX_N):
        self.base_key = base_key
        self.max_n = max_n
    
    def sow(self, index, x):
        return cond_sower(index, named_sow, self.base_key)(x)

    def reap(self, fn, no_reaped=False):
        def fn_args_kwargs(args_kwargs):
            result = fn(*args_kwargs[0], **args_kwargs[1])
            if no_reaped:
                return result, {}
            return result
        def new_fn(*args, **kwargs):
            (results, base_reaped), reaped = call_and_reap(
                fn_args_kwargs,
                (args, kwargs),
                tag="interp",
                allowlist=[
                    self.base_key + f".{i}"
                    for i in range(self.max_n)
                ])
            reaped = {k.lstrip(self.base_key + "."): v for k, v in reaped.items()}
            return results, base_reaped | {self.base_key: reaped}
        return new_fn


post_double_reaper = Reaper("double.resid")
post_single_reaper = Reaper("single.resid")

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
