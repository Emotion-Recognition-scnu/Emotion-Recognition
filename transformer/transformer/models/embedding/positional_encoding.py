"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

#注释
""" 


这个 `positional_encoding.py` 文件定义了一个名为 `PositionalEncoding` 的类，它是神经网络中的一个模块，专门用于计算位置编码（positional encoding）。在变换器（Transformer）模型中，位置编码是一个关键的组成部分，用来给模型提供关于单词在序列中位置的信息。下面详细解释这个类的组成结构、每个部分的作用、实现逻辑，以及参数的传递和存储。

### 类的目的和作用

位置编码被用于变换器模型中，以使模型能够理解序列中单词的顺序。由于变换器模型中的自注意力机制本身不具有处理序列顺序的能力，位置编码通过向每个输入的嵌入向量中添加唯一的位置信息来解决这个问题。

### 构造函数 `__init__`

#### 参数解释

1. **d_model (模型维度)**: 这代表每个词嵌入向量的维度。位置编码的维度需要与词嵌入的维度相同，以便将二者相加。

2. **max_len (最大序列长度)**: 这定义了能被模型处理的最大序列长度。位置编码需要为每个可能的位置生成一个唯一的编码。

3. **device**: 指定计算和存储位置编码的硬件设备（如CPU或GPU）。

#### 实现细节

- 创建一个大小为 `max_len x d_model` 的零矩阵 `self.encoding` 作为位置编码的基础。这个矩阵的大小与输入矩阵相同，以便于后续与词嵌入矩阵相加。

- 使用 `torch.arange` 生成一个从0到 `max_len` 的位置序列，然后将其转换为浮点数并扩展维度，以表示单词在序列中的位置。

- 计算正弦和余弦函数的参数。这里使用 `torch.arange` 生成一个从0到 `d_model` 的序列，步长为2。这是为了分别计算位置编码矩阵的偶数和奇数索引。

- 最后，使用正弦和余弦函数分别填充 `self.encoding` 的偶数和奇数索引列。这样，每个位置的编码都是唯一的，并包含一定的模式以帮助模型识别位置信息。

### 前向传播函数 `forward`

#### 参数解释

- **x**: 输入张量，其大小为 `[batch_size, seq_len]`。

#### 实现细节

- 函数首先确定输入张量的批次大小和序列长度。

- 然后，它从预先计算的 `self.encoding` 中截取与输入序列长度相匹配的部分，并将其返回。这个被截取的位置编码将会被加到输入的词嵌入向量上。

### 与其他代码部分的联系

`PositionalEncoding` 类在变换器模型的编码器和解码器部分中被使用。在模型的每个输入（无论是编码器还是解码器）中，位置编码被加到词嵌入向量上，从而提供关于单词位置的信息。这确保了模型能够利用序列中单词的顺序信息，这对于理解语言至关重要。

总结来说，`PositionalEncoding` 类通过生成唯一的、基于位置的编码，使得变换器模型能够处理序列数据并理解单词间的相对或绝对位置。这对于模型的性能至关重要，尤其是在处理那些依赖于序列顺序的任务时（如文本翻译

或问答系统）。


 """