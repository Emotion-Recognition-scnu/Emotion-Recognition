"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
    
#注释
""" 

ScaleDotProductAttention类。这个类实现了缩放点积注意力机制，是Transformer模型中的一个关键部分。

__init__方法：这个方法初始化类的实例。它首先调用父类的初始化方法，然后定义一个Softmax层，用于在后面的计算中将注意力得分转换为概率分布。

forward方法：这个方法实现了缩放点积注意力的计算过程。它接收四个参数，q（查询），k（键），v（值）和mask（可选的掩码）。

步骤1：计算查询和键的点积，然后除以sqrt(d_tensor)进行缩放。这是为了防止点积的结果过大，导致softmax函数的梯度过小。

步骤2：如果提供了掩码，那么在计算注意力得分之前，会将掩码加到点积的结果上。这是为了在计算注意力得分时忽略某些位置。

步骤3：通过softmax函数将注意力得分转换为概率分布。

步骤4：用注意力得分对值进行加权求和，得到输出。

 """