import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import re
import pickle

class Node(object):
    def __init__(self, num, head, label):
        self.num = num
        self.head = head
        self.label = label


def write_oracle(buffer, buffer_child_to_head_dict):
    arcs = []
    stack = []
    while len(buffer) > 0 or len(stack) != 1:
        if len(stack) > 1 and stack[-1].num == stack[-2].head:
            arcs.append('REDUCE-LEFT-ARC(' + stack[-1].label + ')')
            stack.pop(-2)
        elif len(stack) > 1 and stack[-2].num == stack[-1].head:
            if stack[-1].num in buffer_child_to_head_dict.values():
                arcs.append('SHIFT')
                del buffer_child_to_head_dict[buffer[0].num]
                stack.append(buffer.pop(0))
            else:
                arcs.append('REDUCE-RIGHT-ARC(' + stack[-2].label + ')')
                stack.pop(-1)
        elif len(buffer) > 0:
            arcs.append('SHIFT')
            del buffer_child_to_head_dict[buffer[0].num]
            stack.append(buffer.pop(0))
        else:
            # stack에 독립 item 존재
            break
    return arcs

def set_forget_bias(lstm, num):
    for names in lstm._all_weights:
        for name in filter(lambda n: "bias" in n, names):
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(num)

def lstm_init_uniform_weights(lstm, scale):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                init.uniform_(lstm.__getattr__(p), 0.0, scale)

def linear_init(l, scale):
    l.weight.data.uniform_(0.0, scale)
    l.bias.data.fill_(0)






























