from contextlib import contextmanager
import jax.experimental
from copy import copy
from oryx.core.interpreters.harvest import call_and_reap, sow_cond
import jax.numpy as jnp
import jax


# NOT THREAD SAFE
class Reaper(object):
    def __init__(self, base_key):
        self.base_key = base_key
        self.restrict_to_layers = []
        
    @property
    def buffer_size(self):
        return len(self.restrict_to_layers)
    
    def sow(self, index, x, key=""):
        if self.buffer_size is None or self.buffer_size <= 0:
            return x
        if not isinstance(x, jax.typing.ArrayLike):
            if isinstance(x, dict):
                return {k: self.sow(index, v, key + f".{k}") for k, v in x.items()}
            else:
                raise ValueError(f"Cannot sow {type(x)}")
        prev_clobbered = sow_cond(jnp.empty((self.buffer_size,) + x.shape, dtype=x.dtype), jnp.zeros((), dtype=jnp.bool_), tag=self.tag, name=self.base_key + key)
        allowed_indices = jnp.array(self.restrict_to_layers, dtype=jnp.int32)
        new_clobbered = prev_clobbered.at[jnp.argmax(allowed_indices == index)].set(x)
        sow_cond(new_clobbered, jnp.isin(index, allowed_indices)[0], tag=self.tag, name=self.base_key + key, mode="cond_clobber")
        return x
    
    @property
    def tag(self):
        return self.base_key + ".interp"

    @contextmanager
    def reaping(self, *layers):
        prev_restrict_to_layers = self.restrict_to_layers
        self.restrict_to_layers = layers
        yield self
        self.restrict_to_layers = prev_restrict_to_layers

    def reap(self, fn, no_reaped=False, restrict_to_layers=[]):
        def fn_args_kwargs(args_kwargs):
            result = fn(*args_kwargs[0], **args_kwargs[1])
            if no_reaped:
                return result, {}
            return result
        def new_fn(*args, **kwargs):
            with self.reaping(*restrict_to_layers):
                (results, base_reaped), reaped = call_and_reap(
                    fn_args_kwargs,
                    tag=self.tag)(
                        (args, kwargs)
                    )
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
