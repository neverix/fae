from transformers import AutoTokenizer
from .modelling_t5 import FlaxT5EncoderModel
from functools import partial
from loguru import logger
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
from .quant_loading import load_thing, save_thing
import numpy as np
import jax
import qax

from .quant import quantize_matrix, QuantMatrix

jit_quantize = jax.jit(quantize_matrix, static_argnames=("use_approx", "group_size"))


def maybe_quantize(path, param, mesh_and_axis=None):
    will_skip = (
        param.ndim != 2
        or "embed" in jax.tree_util.keystr(path)
        or any(d > 11_000 for d in param.shape)
    )
    if will_skip:
        return param
    param = quantize_matrix(param, use_approx=True, group_size=32)
    return param


def quantize_params_tree(params, **kwargs):
    return jax.tree_util.tree_map_with_path(partial(maybe_quantize, **kwargs), params)


def to_device(param, mesh_and_axis=None):
    if isinstance(param, QuantMatrix):
        return param.with_mesh_and_axis(mesh_and_axis)
    return jax.device_put(param, jax.sharding.NamedSharding(
        mesh_and_axis[0],
        jax.sharding.PartitionSpec(*((None,) * param.ndim))))

def to_device_params_tree(params, **kwargs):
    return jax.tree.map(
        partial(to_device, **kwargs),
        params,
        is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue),
    )

@partial(jax.jit, static_argnums=(0,))
def run_model(wrapped_model, params, ids):
    return wrapped_model(params=params, input_ids=ids).last_hidden_state

class T5EncoderInferencer(object):
    def __init__(self, mesh, model_name=None):
        if model_name is None:
            model_name = "nev/t5-v1_1-xxl-flax"
        self.mesh = mesh
        self.input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantized_path = Path("somewhere/quantized_t5")
        logger.info("Loading model")
        model, params = FlaxT5EncoderModel.from_pretrained(
            model_name, _do_init=False, dtype=jnp.bfloat16
        )
        if not quantized_path.exists():
            logger.info("Quantizing model")
            params = quantize_params_tree(params)
            logger.info("Saving quantized model")
            save_thing(params, quantized_path)
        else:
            params = load_thing(quantized_path)
            predicate = lambda x: isinstance(x, dict) and "quants" in x and "scales" in x
            def quantify(param):
                if predicate(param):
                    return QuantMatrix(
                        **param,
                        orig_dtype=jnp.bfloat16,
                        use_approx=True,
                        use_kernel=True,
                        mesh_and_axis=None,
                    )
                return param
            params = jax.tree.map(quantify, params, is_leaf=lambda x: predicate(x) or isinstance(x, qax.primitives.ArrayValue))
        logger.info("Moving model to device")
        params = to_device_params_tree(params, mesh_and_axis=(mesh, None))
        
        def set_use_kernel(tree, value):
            def op(x):
                if isinstance(x, QuantMatrix):
                    x.use_kernel = value

            jax.tree.map(op, tree, is_leaf=lambda x: isinstance(x, QuantMatrix))

        set_use_kernel(params, True)

        self.wrapped_model = eqx.filter_jit(qax.use_implicit_args(model.__call__))
        self.params = params

    def __call__(self, texts):
        encoder_input = self.create_inputs(texts)
        return run_model(self.wrapped_model, self.params, encoder_input)

    def create_inputs(self, texts):
        encoder_inputs = []
        with jax.default_device(jax.devices("cpu")[0]):
            for text in texts:
                encoder_input = jnp.asarray(self.tokenizer.encode(text), dtype=jnp.int32)[
                    None
                ]
                encoder_input = jnp.pad(
                    encoder_input,
                    ((0, 0), (0, 512 - encoder_input.shape[-1])),
                    mode="constant",
                    constant_values=self.tokenizer.pad_token_id,
                )
                encoder_inputs.append(encoder_input)
        return self.shard_inputs(jnp.concatenate(encoder_inputs, axis=0))

    def shard_inputs(self, inputs):
        return jax.device_put(
            inputs.reshape(min(self.mesh.shape["dp"], inputs.shape[0]), -1, inputs.shape[-1]), self.input_sharding
        )


if __name__ == "__main__":
    import jax_smi
    jax_smi.initialise_tracking(7)
    mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 4, 1), ("dp", "fsdp", "tp"))
    inferencer = T5EncoderInferencer(mesh)
    print(inferencer(["Hello, world!"] * 8).shape)

