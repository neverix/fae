from transformers import FlaxAutoModelForSeq2SeqLM, FlaxT5EncoderModel, AutoTokenizer
import jax.numpy as jnp
import numpy as np
import jax
import qax

from .quant import quantize_matrix, QuantMatrix

jit_quantize = jax.jit(quantize_matrix, static_argnames=("use_approx", "group_size"))


def maybe_quantize(path, param):
    if param.ndim != 2:
        return param

    if "embed" in jax.tree_util.keystr(path):
        # Don't want to relative_attention_bias embedding
        return param

    # Avoid embedding tables/final projection
    if any(d > 9000 for d in param.shape):  #
        return param
    return quantize_matrix(param, use_approx=True, group_size=32)


def quantize_params_tree(params):
    return jax.tree_util.tree_map_with_path(maybe_quantize, params)

if __name__ == "__main__":
    model_name = "nev/t5-v1_1-xxl-flax"
    model, params = FlaxT5EncoderModel.from_pretrained(model_name, _do_init=False, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantized_params = quantize_params_tree(params)

    quantized_params = jax.device_put(quantized_params, jax.devices("tpu")[0])
    params = jax.device_put(params, jax.devices("tpu")[0])
    
    wrapped_model = jax.jit(qax.use_implicit_args(model.__call__))


    def set_use_kernel(tree, value):
        def op(x):
            if isinstance(x, QuantMatrix):
                x.use_kernel = value

        jax.tree.map(op, tree, is_leaf=lambda x: isinstance(x, QuantMatrix))


    input = "The color of the sky is: "
    encoder_input = jnp.asarray(tokenizer.encode(input), dtype=jnp.int32)[None]
    decoder_start = jnp.asarray([[tokenizer.pad_token_id]], dtype=jnp.int32)

    encoder_input = encoder_input
    decoder_start = decoder_start

    kwargs = {"input_ids": encoder_input}
    set_use_kernel(quantized_params, True)
    quantized_logits = wrapped_model(params=quantized_params, **kwargs).last_hidden_state
    set_use_kernel(quantized_params, False)
    quantized_logits_no_kernel = wrapped_model(params=quantized_params, **kwargs).last_hidden_state
    base_logits                = wrapped_model(params=params, **kwargs).last_hidden_state
    print(jnp.mean(jnp.abs(quantized_logits - quantized_logits_no_kernel)), jnp.mean(jnp.abs(quantized_logits_no_kernel)))
    print(jnp.mean(jnp.abs(base_logits - quantized_logits)), jnp.mean(jnp.abs(base_logits)))
    exit()
