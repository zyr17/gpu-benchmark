#!/usr/bin/env python3
import torch
from tqdm import tqdm
import time
from torch import nn
import sys


class lstm_model(nn.Module):
    def __init__(self, input, hidden, layer_number):
        super().__init__()
        self.input = input
        self.hidden = hidden
        self.layer_number = layer_number
        self.lstm = nn.LSTM(input, hidden, layer_number, batch_first = True)

    def forward(self, x, h = None, c = None):
        output, (h, c) = self.lstm(x)
        return output, (h, c)


@torch.no_grad()
def test(dtype, bs, num_iters):
    seq_len = 100
    model = lstm_model(256, 256, 1).cuda().type(dtype)
    x = torch.randn(bs, seq_len, 256).cuda().type(dtype)
    start_time = time.time()
    _ = model(x)  # first time not count
    for _ in tqdm(range(num_iters)):
        output, (h, c) = model(x)  # noqa
    end_time = time.time()
    print(f'{dtype} bs={bs} {num_iters}it LSTM inference time:', end_time - start_time)
    print(
        f'{dtype} bs={bs} {num_iters}it LSTM inference memory usage:', 
        torch.cuda.memory_reserved()
    )
    torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == '16':
        test(torch.float16, 24576, 2000)
    elif sys.argv[1] == 'b16':
        test(torch.bfloat16, 24576, 2000)
    elif sys.argv[1] == '32':
        test(torch.float32, 12288, 2000)
    elif sys.argv[1] == '64':
        test(torch.float64, 6144, 100)
