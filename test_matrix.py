#!/usr/bin/env python3
import sys
import torch
from tqdm import tqdm
import time


def test(dtype, num_iters):
    number = 32768
    a = torch.rand((number, number), device = 'cuda').type(dtype)
    res = 0
    _ = a.matmul(a).sum().item()
    start_time = time.time()
    for _ in tqdm(range(num_iters)):
        res += a.matmul(a).sum().item()
    print(
        f'{dtype} sz=32768 {num_iters}it matrix multiply time:', 
        time.time() - start_time
    )
    print(
        f'{dtype} sz=32768 {num_iters}it matrix multiply memory usage:', 
        torch.cuda.memory_reserved()
    )


if __name__ == '__main__':
    if sys.argv[1] == '16':
        test(torch.float16, 200)
    elif sys.argv[1] == 'b16':
        test(torch.bfloat16, 200)
    elif sys.argv[1] == '32':
        test(torch.float32, 100)
    elif sys.argv[1] == '64':
        test(torch.float64, 5)
