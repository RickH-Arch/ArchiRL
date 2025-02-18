import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

    #计算输入数据经过卷积层处理后输出元素的总数
    def _get_conv_out(self,shape):
        o = self.conv(torch.zeros(1,*shape))#这种调用方式必须呈批次输入
        return int(np.prod(o.size()))#计算所有维度大小的乘积
    
    def forward(self,x):#接受4D张量，1:批大小、2:颜色通道、3/4:图像尺寸
        conv_out = self.conv(x).view(x.size()[0],-1)
        return self.fc(conv_out)

