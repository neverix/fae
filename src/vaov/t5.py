from transformers import AutoTokenizer
from .modelling_t5 import FlaxT5EncoderModel
from functools import partial
import jax.numpy as jnp
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
    if mesh_and_axis and will_skip:
        return jax.device_put(param,
                                jax.sharding.NamedSharding(mesh_and_axis[0],
                                                           jax.sharding.PartitionSpec(*((None,) * param.ndim))))
    if will_skip:
        return param
    param = quantize_matrix(param, use_approx=True, group_size=32)
    if mesh_and_axis:
        param = param.with_mesh_and_axis(mesh_and_axis)
    return param


def quantize_params_tree(params, **kwargs):
    return jax.tree_util.tree_map_with_path(partial(maybe_quantize, **kwargs), params)

if __name__ == "__main__":
    import jax_smi
    jax_smi.initialise_tracking(7)
    mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 4, 1), ("dp", "fsdp", "tp"))
    input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))

    def shard_inputs(inputs):
        return jax.device_put(
            inputs.reshape(mesh.shape["dp"], -1, inputs.shape[-1]), input_sharding
        )

    model_name = "nev/t5-v1_1-xxl-flax"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input = "The color of the sky is: "
    encoder_input = jnp.asarray(tokenizer.encode(input), dtype=jnp.int32)[None]
    encoder_input = jnp.pad(
        encoder_input,
        ((0, 0), (0, 128 - encoder_input.shape[-1])),
        mode="constant",
        constant_values=tokenizer.pad_token_id,
    )
    encoder_input = jnp.repeat(encoder_input, 256, axis=0)
    encoder_input = shard_inputs(encoder_input)

    model, params = FlaxT5EncoderModel.from_pretrained(model_name, _do_init=False, dtype=jnp.bfloat16)
    params = quantize_params_tree(params, mesh_and_axis=(mesh, None))

    wrapped_model = jax.jit(qax.use_implicit_args(model.__call__))


    def set_use_kernel(tree, value):
        def op(x):
            if isinstance(x, QuantMatrix):
                x.use_kernel = value

        jax.tree.map(op, tree, is_leaf=lambda x: isinstance(x, QuantMatrix))

    set_use_kernel(params, True)
    quantized_states = jax.jit(lambda params, ids: wrapped_model(params=params, input_ids=ids).last_hidden_state)(params, encoder_input)
    print(jax.jit(lambda x: jnp.mean(jnp.abs(x)))(quantized_states))
    exit()
