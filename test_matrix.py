#!/usr/bin/env python3
import sys
import torch
from tqdm import tqdm
import time


def test(dtype):
    number = 32768
    a = torch.rand((number, number), device = 'cuda').type(dtype)
    res = 0
    start_time = time.time()
    _ = a.matmul(a).sum().item()
    for _ in tqdm(range(100)):
        res += a.matmul(a).sum().item()
    print(f'{dtype} sz=32768 matrix multiply time:', time.time() - start_time)
    print(
        f'{dtype} sz=32768 matrix multiply memory usage:', torch.cuda.memory_reserved())


if __name__ == '__main__':
    if sys.argv[1] == '16':
        test(torch.float16)
    elif sys.argv[1] == '32':
        test(torch.float32)
    elif sys.argv[1] == '64':
        test(torch.float64)
