# Benchmark for 24G VRAM GPUs

Test performance of 24GB VRAM GPUs, i.e. 3090, 3090Ti, 4090, 4090D, 7900XTX.

Disclamer: This repo is just be built because of curiosity, as the author has chance
to get in touch with (probably) all 5 models of 24GB VRAM consumer GPUs (before mid of 
2024). Author is not familiar with LLMs and Stable Diffusions, and just borrow the 
popular codes and modified them to be able to run. No guarantee for result reliability.

For regions that cannot connect to huggingface, you can try to download the model by 
yourself, and set offline mode to run the experiments, by setting the following 
environments:
```
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_OFFLINE=1
```

The provided `docker-compose.yml` uses above environments, if you want to download them
from huggingface, uncomment them. It builds from pytorch2.3.0-cuda11.8 for compatibility
reasons, if you want to test on newer version of pytorch or cuda, you can build by 
yourself.

The docker is used for Nvidia cards, and for AMD cards (i.e. 7900XTX), you should build
by yourself.

## install environments

Based on Python, PyTorch and huggingface libraries.

Install by: `pip install -r requirements.txt`.

You need to collect model weights from original (if you can connect to the model hub)
or download the model manually to your `~/.local/` folder. When you are using docker,
you need to mount them into the docker.

We test the methods with different precisions, batch sizes, and iteration rounds.
Batch sizes will be set to use approximately greater than half GPU memory.
Iteration is determined by the time used for one iteration, each experiment will be 
controlled to run about 1 to 5 minutes on 4090.
`PRECISION` may be 4, 8, 16, b16, 32 or 64, for different tests. The availabilities 
are:

| precision | `matrix` | `resnet` | `LSTM` | `Stable Diffusion` | `Alpaca Lora` |
| ------- | ------- | ------- | ------- | ------- | ------- |
| FP4 | - | - | - | - | √ |
| FP8 | - | - | - | - | √ |
| FP16 | √ | √ | √ | √ | √ |
| BF16 | √ | √ | √ | √ | - |
| FP32 | √ | √ | √ | √ | - |
| FP64 | √ | √ | √ | - | - |

We find BF16 has almost same calculation speed but will consume more GPU memory than 
FP16, and do not know why. This makes the Alpaca Lora model unable to run with BF16,
and other models should decrease the batch size to successfully run.

## Training benchmarks

run `train_xxx.xx PRECISION`. 

or `train_all.sh` to test all.

## Inference benchmarks

run `inference_xxx.xx`

or `inference_all.sh` to inference all.

## Other benchmarks

Now only `test_matrix.py` to do vanilla matrix-matrix product.

## Code detail

After test is done, it will print the time cost and VRAM usage by `time.time` and 
`torch.cuda.memory_reserved`.

## Testing environments for each cards

This is not a exact test, many different variables exist for each cards (We do not have
time to remove card out of its original case and put them on a same machine). Here 
records the detailed hardware settings, OS information and driver information.

TBD

### 3090

### 3090 Ti

### 4090

### 4090 D

### 7900 XTX
