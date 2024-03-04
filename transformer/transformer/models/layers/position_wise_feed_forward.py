"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
#注释
""" 
PositionWiseFeedForward类。这个类实现了位置级别的前馈神经网络，也是Transformer模型中的一个关键部分。

__init__方法：这个方法初始化类的实例。它首先调用父类的初始化方法，然后定义两个线性层和一个ReLU激活函数。这两个线性层和ReLU激活函数构成了一个前馈神经网络。

forward方法：这个方法实现了前馈神经网络的计算过程。它接收一个参数，x（输入）。

步骤1：通过第一个线性层进行线性变换。

步骤2：通过ReLU激活函数进行非线性变换。

步骤3：通过第二个线性层进行线性变换，得到输出。

这两个类的主要作用是对输入数据进行处理，以便更好地捕捉数据中的模式。在Transformer模型中，缩放点积注意力机制可以帮助模型同时关注输入序列的不同位置，而位置级别的前馈神经网络则可以帮助模型学习输入数据的内部表示。

 """