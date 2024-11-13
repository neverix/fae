from fire import Fire
from datasets import load_dataset
from more_itertools import chunked
from .ensemble import FluxEnsemble


def main():
    prompts_dataset = load_dataset("k-mktr/improved-flux-prompts")
    prompts_iterator = prompts_dataset["train"]["prompt"]
    ensemble = FluxEnsemble(use_schnell=True, use_fsdp=True)
    for prompts in chunked(prompts_iterator, 32):
        images, reaped = ensemble.sample(prompts, debug_mode=True, decode_latents=False, sample_steps=1)
        print(images.shape)
        print({k: v.shape for k, v in reaped.items()})
        for v in reaped.values():
            v.delete()


if __name__ == "__main__":
    Fire(main)