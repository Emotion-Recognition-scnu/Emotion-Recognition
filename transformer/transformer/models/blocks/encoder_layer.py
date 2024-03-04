"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
#注释
""" 

这段代码定义了一个名为 `EncoderLayer` 的类，该类是神经网络变换器模型中编码器层的实现。它继承自 PyTorch 的 `nn.Module`，使其成为一个可用于构建更大型神经网络模型的模块。下面是对这个类的构造函数和前向传播函数的详细解释：

### 构造函数 `__init__`

#### 目的和作用
构造函数的主要目的是初始化编码器层需要的各个子模块。这些子模块共同协作，以完成对输入数据的变换。

#### 参数解释
1. **d_model (模型维度)**: 这是每个输入词嵌入的维度。在变换器模型中，所有的子层和嵌入层都使用相同的维度`d_model`。

2. **ffn_hidden (前馈神经网络隐藏层大小)**: 这定义了编码器层内的前馈网络的隐藏层维度。这个维度通常大于`d_model`，允许网络在内部处理更高维度的数据，从而捕获更复杂的特征。

3. **n_head (注意力头数量)**: 指定在多头注意力机制中使用的头数。多头注意力允许模型在不同的表示子空间中并行捕获信息。

4. **drop_prob (丢弃概率)**: 指定在dropout层中使用的丢弃概率，这有助于防止模型过拟合。

#### 实现细节
- **多头注意力层 (`self.attention`)**: 用于计算输入数据的自注意力。这个步骤允许每个输入位置都能考虑到其他位置的信息，从而捕捉序列内部的关联关系。
- **层归一化 (`self.norm1` 和 `self.norm2`)**: 用于标准化输入，有助于稳定训练过程并加快收敛速度。
- **丢弃层 (`self.dropout1` 和 `self.dropout2`)**: 在模型训练时随机丢弃一部分特征，以防止过拟合。
- **位置前馈网络 (`self.ffn`)**: 一个全连接的前馈网络，为模型增加了额外的非线性，能够捕获更复杂的特征。

### 前向传播函数 `forward`

#### 目的和作用
`forward`方法定义了数据通过编码器层时的处理流程。它规定了如何将输入数据转换为下一层（或最终输出）的形式。

#### 参数解释
1. **x (输入数据)**: 这是传入编码器层的数据，通常是经过嵌入层处理的序列数据。
2. **src_mask (源掩码)**: 用于掩蔽（屏蔽）序列中的某些部分，通常用于遮盖掉填充的位置，以防止这些位置影响自注意力的计算。

#### 实现细节
- **自注意力计算**: 使用输入`x`作为查询（query）、键（key）和值（value）来计算自注意力，同时应用源掩码。
- **添加和归一化**: 通过dropout正则化自注意力的输出，然后将其与原始输入相加，接着应用层归一化。
- **位置前馈网络**: 将自注意力的输出通过前馈网络进行进一步处理。
- **再次添加和归一化**: 对前馈网络的输出进行相同的处理，即dropout、加原始输入、层归一化。

### 与其他部分的代码联系

`EncoderLayer` 类是变换器模型中多个

编码器层堆叠在一起的一部分。在完整的变换器模型中，多个这样的编码器层被顺序堆叠，每层的输出都作为下一层的输入。这种设计允许模型捕获和处理复杂的序列数据特征，特别是在处理诸如自然语言理解等复杂任务时。每个编码器层通过其自注意力机制和前馈网络，对输入数据进行变换和提炼，为解码器层提供精炼的信息表示。

 """