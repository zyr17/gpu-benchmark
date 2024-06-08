#!/usr/bin/env python3
import sys
import torch
from torch import autocast
import time
from diffusers import StableDiffusionPipeline


def test(dtype, bs, steps):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        use_auth_token=True,
        torch_dtype = dtype,
    ).to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    start_time = time.time()
    with autocast("cuda"):
        _ = pipe([prompt] * bs, num_inference_steps = steps)["images"]
    print(
        f'{dtype} bs=10 Stable Diffusion 10it inference time:', 
        time.time() - start_time
    )
    print(
        f'{dtype} bs=10 Stable Diffusion 10it inference memory usage:', 
        torch.cuda.memory_reserved()
    )


if __name__ == '__main__':
    if sys.argv[1] == '32':
        test(torch.float32, 10, 500)
    elif sys.argv[1] == '16':
        test(torch.float16, 10, 500)
    elif sys.argv[1] == 'b16':
        test(torch.bfloat16, 10, 500)
