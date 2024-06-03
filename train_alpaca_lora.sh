#!/bin/bash
cd alpaca-lora
DTYPE=$1 python finetune.py --base_model 'luodian/llama-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca'
