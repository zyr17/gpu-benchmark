#!/usr/bin/env python3
import torch
from tqdm import tqdm
import time
from torch import nn
import sys


class resnet(nn.Module):
    def __init__(self, input, hidden, layer_number):
        super().__init__()
        self.input = input
        self.hidden = hidden
        self.layer_number = layer_number
        self.lstm = nn.LSTM(input, hidden, layer_number, batch_first = True)

    def forward(self, x, h = None, c = None):
        output, (h, c) = self.lstm(x)
        return output, (h, c)


def test(dtype, bs, num_iters):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model = model.type(dtype).cuda()
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    input = torch.randn(bs, 3, 224, 224).cuda().type(dtype)
    label = torch.randint(0, 1000, (bs,)).cuda()
    _ = model(input)  # first time not count
    start_time = time.time()
    for _ in tqdm(range(num_iters)):
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
    end_time = time.time()
    print(f'{dtype} bs={bs} {num_iters}it ResNet152 train time:', end_time - start_time)
    print(
        f'{dtype} bs={bs} {num_iters}it ResNet152 train memory usage:', 
        torch.cuda.memory_reserved()
    )
    torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == '16':
        test(torch.float16, 192, 300)
    elif sys.argv[1] == 'b16':
        test(torch.bfloat16, 160, 300)
    elif sys.argv[1] == '32':
        test(torch.float32, 96, 300)
    elif sys.argv[1] == '64':
        test(torch.float64, 48, 50)
