from transformers import AutoTokenizer
from .modelling_t5 import FlaxT5EncoderModel
from functools import partial
from loguru import logger
import jax.numpy as jnp
import equinox as eqx
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
    return param

def to_device_params_tree(params, **kwargs):
    return jax.tree_util.tree_map_with_path(partial(to_device, **kwargs), params)

if __name__ == "__main__":
    import jax_smi
    jax_smi.initialise_tracking(7)
    mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 4, 1), ("dp", "fsdp", "tp"))
    input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))

    def shard_inputs(inputs):
        return jax.device_put(
            inputs.reshape(mesh.shape["dp"], -1, inputs.shape[-1]), input_sharding
        )

    logger.info("Creating inputs")
    model_name = "nev/t5-v1_1-xxl-flax"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = "A mystic cat with a sign that says hello world!"
    encoder_input = jnp.asarray(tokenizer.encode(input), dtype=jnp.int32)[None]
    encoder_input = jnp.pad(
        encoder_input,
        ((0, 0), (0, 512 - encoder_input.shape[-1])),
        mode="constant",
        constant_values=tokenizer.pad_token_id,
    )
    encoder_input = jnp.repeat(encoder_input, 16, axis=0)
    encoder_input = shard_inputs(encoder_input)

    logger.info("Loading model")
    model, params = FlaxT5EncoderModel.from_pretrained(model_name, _do_init=False, dtype=jnp.bfloat16)
    logger.info("Quantizing model")
    params = quantize_params_tree(params)
    logger.info("Moving model to device")
    params = to_device_params_tree(params, mesh_and_axis=(mesh, None))

    wrapped_model = eqx.filter_jit(qax.use_implicit_args(model.__call__))


    def set_use_kernel(tree, value):
        def op(x):
            if isinstance(x, QuantMatrix):
                x.use_kernel = value

        jax.tree.map(op, tree, is_leaf=lambda x: isinstance(x, QuantMatrix))

    set_use_kernel(params, True)
    logger.info("Running model")
    quantized_states = jax.jit(lambda params, ids: wrapped_model(params=params, input_ids=ids).last_hidden_state)(params, encoder_input)
    logger.info("Saving outputs")
    np.save("somewhere/quantized_states.npy", np.asarray(quantized_states.astype(np.float32)))
