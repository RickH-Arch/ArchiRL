import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

# distributional DQN params
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init = 0.017, bias = True):
        super(NoisyLinear, self).__init__(in_features,out_features,bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)

        #创建零张量
        z = torch.zeros(out_features, in_features)
        #将张量z注册为缓冲区， 缓冲区不会被视为模型的可训练参数，但会作为模型状态的一部分保存和加载。
        #这个缓冲区用于存储权重噪声
        self.register_buffer("epsilon_weight",z)

        if bias:
            w = torch.full((out_features,),sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias",z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std,std)
        self.bias.data.uniform_(-std,std)


# forward 方法的核心功能是在标准的线性变换基础上引入噪声。
# 每次调用 forward 方法时，都会重新生成权重和偏置的噪声，
# 从而使模型在训练过程中具有一定的随机性，这种随机性可以提高模型的探索能力，
# 尤其在强化学习等领域有重要应用。
# 通过可训练的噪声强度参数 self.sigma_weight 和 self.sigma_bias，模型可以自动调整噪声的影响程度。
    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        v = self.weight + self.sigma_weight * self.epsilon_weight.data
        return F.linear(input, v, bias)