from contextlib import contextmanager
from functools import partial
import jax.experimental
from copy import copy
from oryx.core.interpreters.harvest import call_and_reap, sow_cond, sow, plant
import jax.numpy as jnp
import jax


# NOT THREAD SAFE
class Reaper(object):
    def __init__(self, base_key):
        self.base_key = base_key
        self.buffer_size = None
    
    def sow(self, index, x):
        if self.buffer_size is None or self.buffer_size <= 0:
            return x
        if not isinstance(x, jax.typing.ArrayLike):
            return jax.tree.map(partial(self.sow, index), x)
        allowed_indices = sow(jnp.full(self.buffer_size, -1, dtype=jnp.int32), tag=self.tag, name="allowed_indices", mode="clobber")
        prev_clobbered = sow_cond(jnp.empty((self.buffer_size,) + x.shape, dtype=x.dtype), jnp.zeros((), dtype=jnp.bool_), tag=self.tag, name=self.base_key)
        new_clobbered = prev_clobbered.at[jnp.argmax(allowed_indices == index)].set(x)
        sow_cond(new_clobbered, jnp.isin(index, allowed_indices), tag=self.tag, name=self.base_key, mode="cond_clobber")
        return x
    
    @property
    def tag(self):
        return self.base_key + ".interp"

    def reap(self, fn, no_reaped=False, restrict_to_layers=[]):
        def fn_args_kwargs(args_kwargs):
            result = fn(*args_kwargs[0], **args_kwargs[1])
            if no_reaped:
                return result, {}
            return result
        def new_fn(*args, **kwargs):
            prev_buffer_size = self.buffer_size
            self.buffer_size = len(restrict_to_layers)
            allowed_indices = jnp.array(restrict_to_layers, dtype=jnp.int32)
            (results, base_reaped), reaped = call_and_reap(
                plant(fn_args_kwargs, tag=self.tag),
                tag=self.tag,
                allowlist=[self.base_key])(
                    {"allowed_indices": allowed_indices},
                    (args, kwargs)
                )
            self.buffer_size = prev_buffer_size
            return results, base_reaped | reaped
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
