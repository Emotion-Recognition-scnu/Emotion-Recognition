"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
""" 

这个 `token_embedding.py` 文件定义了一个名为 `TokenEmbedding` 的类，它是一个用于生成词嵌入的神经网络模块。此类继承自 PyTorch 的 `nn.Embedding`，专门用于将词汇表中的词转换为密集向量表示。以下是对这个类的构造函数和整体结构的详细解释。

### 类的目的和作用

`TokenEmbedding` 类的主要作用是将词（通常表示为整数索引）转换为固定维度的嵌入向量。这在自然语言处理（NLP）中非常常见，尤其是在使用神经网络处理文本数据时。词嵌入向量能够捕获词之间的语义信息，并以一种神经网络可以有效处理的方式表示这些信息。

### 构造函数 `__init__`

#### 参数解释

1. **vocab_size (词汇表大小)**: 这表示词汇表中的词的数量。每个唯一的词都会被分配一个索引，`vocab_size` 就是这些索引的总数。

2. **d_model (模型维度)**: 这代表每个词嵌入向量的维度。所有的词都会被转换为这个维度的向量。

#### 实现细节

- 类通过调用 `super(TokenEmbedding, self).__init__` 初始化其父类 `nn.Embedding`。`nn.Embedding` 是 PyTorch 中的一个预定义类，专门用于创建一个可以学习的词嵌入矩阵。

- 在这个初始化过程中，`vocab_size` 和 `d_model` 被传递给 `nn.Embedding`，它们分别定义了嵌入矩阵的行数（每个词对应一行）和列数（每个词的嵌入向量维度）。

- `padding_idx=1` 是一个额外的参数，它指定索引为 1 的词（通常用于表示填充）的嵌入应该是零向量。这在处理长度不一的序列时非常有用，因为它可以让模型忽略填充的部分。

### 与其他代码部分的联系

`TokenEmbedding` 类通常与其他 NLP 模型组件一起使用，比如变换器（Transformer）模型的编码器和解码器。在这些模型中，文本首先被转换为索引序列，然后通过 `TokenEmbedding` 转换为嵌入向量序列。这些嵌入向量之后被用作模型的输入，进行后续的处理，如通过自注意力机制进行特征提取。

总结来说，`TokenEmbedding` 类提供了将词汇表中的词转换为密集向量表示的能力，这是大多数现代 NLP 模型的基础和关键部分。通过这种方式，模型能够学习词之间的复杂关系，并在处理文本任务时使用这些信息。


 """