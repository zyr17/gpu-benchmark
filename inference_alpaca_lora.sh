#!/bin/bash
cd alpaca-lora
DTYPE=$1 python generate.py --load_8bit --base_model 'luodian/llama-7b-hf' --lora_weights 'tloen/alpaca-lora-7b'
