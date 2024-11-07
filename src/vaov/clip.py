from transformers import AutoTokenizer, AutoConfig, FlaxCLIPTextModel
import jax.numpy as jnp
import numpy as np
import torch

if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    text_encodings = torch.load("somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True)
    t5_emb, clip_emb = (x.detach().cpu().float().numpy() for x in text_encodings[:2])
    
    clip_name = "openai/clip-vit-large-patch14"
    tokenizer = AutoTokenizer.from_pretrained(clip_name)
    config = AutoConfig.from_pretrained(clip_name)
    
    prompt = "A mystic cat with a sign that says hello world!"
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=config.text_config.max_position_embeddings,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="np",
    ).input_ids
    
    model, params = FlaxCLIPTextModel.from_pretrained(clip_name, _do_init=False, dtype=jnp.bfloat16)
    outputs = model(text_inputs, params=params).pooler_output
    clip_emb_2 = np.asarray(outputs.astype(np.float32))
    print(
        np.mean(np.abs(clip_emb - clip_emb_2)),
        np.mean(np.abs(clip_emb)),
        np.mean(np.abs(clip_emb_2)),
    )
    print("Model loaded")