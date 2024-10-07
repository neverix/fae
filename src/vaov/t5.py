from transformers import FlaxT5EncoderModel, AutoTokenizer
from functools import partial
import jax.numpy as jnp
import numpy as np
import jax
import qax

from .quant import quantize_matrix, QuantMatrix

jit_quantize = jax.jit(quantize_matrix, static_argnames=("use_approx", "group_size"))


def maybe_quantize(path, param, mesh_and_axis=None):
    if mesh_and_axis:
        param_ = jax.device_put(param,
                                jax.sharding.NamedSharding(mesh_and_axis[0],
                                                           jax.sharding.PartitionSpec(*((None,) * param.ndim))))
    if param.ndim != 2:
        return param_  

    if "embed" in jax.tree_util.keystr(path):
        # Don't want to relative_attention_bias embedding
        return param_

    # Avoid embedding tables/final projection
    if any(d > 9000 for d in param.shape):  #
        return param_
    param = quantize_matrix(param, use_approx=True, group_size=32)
    if mesh_and_axis:
        param = param.with_mesh_and_axis(mesh_and_axis)
    return param


def quantize_params_tree(params, **kwargs):
    return jax.tree_util.tree_map_with_path(partial(maybe_quantize, **kwargs), params)

if __name__ == "__main__":
    mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 4, 1), ("dp", "fsdp", "tp"))
    input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))

    model_name = "nev/t5-v1_1-xxl-flax"
    model, params = FlaxT5EncoderModel.from_pretrained(model_name, _do_init=False, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    params = quantize_params_tree(params, mesh_and_axis=(mesh, None))

    wrapped_model = jax.jit(qax.use_implicit_args(model.__call__))


    def set_use_kernel(tree, value):
        def op(x):
            if isinstance(x, QuantMatrix):
                x.use_kernel = value

        jax.tree.map(op, tree, is_leaf=lambda x: isinstance(x, QuantMatrix))

    def shard_inputs(inputs):
        return jax.device_put(inputs.reshape(mesh.shape["dp"], -1, inputs.shape[1]), input_sharding)

    input = "The color of the sky is: "
    encoder_input = jnp.asarray(tokenizer.encode(input), dtype=jnp.int32)[None]
    encoder_input = jnp.pad(encoder_input, (0, 128 - encoder_input.shape[1]), mode="constant")
    encoder_input = jnp.repeat(encoder_input, 4096, axis=0)
    encoder_input = shard_inputs(encoder_input)

    set_use_kernel(params, True)
    quantized_states = jax.vmap(lambda ids: wrapped_model(params=params, input_ids=ids).last_hidden_state)(encoder_input)
    print(jnp.mean(jnp.abs(quantized_states)))
    exit()
