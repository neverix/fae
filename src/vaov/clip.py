from transformers import AutoTokenizer, AutoConfig, FlaxCLIPTextModel
from loguru import logger
import jax.numpy as jnp
import numpy as np
import torch
import jax

class CLIPInterface:
    def __init__(self, mesh, clip_name="openai/clip-vit-large-patch14"):
        logger.info("Creating CLIP model")
        self.mesh = mesh
        self.input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))

        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)
        self.config = AutoConfig.from_pretrained(clip_name)

        logger.info("Loading CLIP model:", clip_name)
        with jax.default_device(jax.devices("cpu")[0]):
            self.model, params = FlaxCLIPTextModel.from_pretrained(
                clip_name, _do_init=False, dtype=jnp.bfloat16
            )
        params = jax.tree_map(
            lambda x:
                jax.device_put(x,
                               jax.sharding.NamedSharding(
                                   mesh,
                                   jax.sharding.PartitionSpec(*((None,) * x.ndim)))),
                params)
        self.params = params
    
    def __call__(self, texts):
        inputs = self.create_inputs(texts)
        outputs = jax.vmap(lambda i, p: self.model(i, params=p).pooler_output, in_axes=(0, None))(
            inputs, self.params
        )
        return outputs
    
    def create_inputs(self, texts):
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.config.text_config.max_position_embeddings,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="np",
        ).input_ids
        text_inputs = text_inputs.reshape(self.mesh.shape["dp"], -1, text_inputs.shape[-1])
        return jax.device_put(text_inputs, self.input_sharding)

if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    text_encodings = torch.load("somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True)
    t5_emb, clip_emb = (x.detach().cpu().float().numpy() for x in text_encodings[:2])
    
    mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 4, 1), ("dp", "fsdp", "tp"))
    inferencer = CLIPInterface(mesh)
    prompt = "A mystic cat with a sign that says hello world!"
    outputs = inferencer([prompt] * 8)
    clip_emb_2 = np.asarray(outputs.astype(np.float32))[0, :1]
    print(
        np.mean(np.abs(clip_emb - clip_emb_2)),
        np.mean(np.abs(clip_emb)),
        np.mean(np.abs(clip_emb_2)),
    )
    print("Model loaded")