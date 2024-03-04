"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
    
#注释
""" 

这个 `TransformerEmbedding` 类是一个神经网络模块，专为变换器（Transformer）模型中的嵌入层设计。它结合了词嵌入（token embedding）和位置编码（positional encoding），以提供对文本数据的有效表示。以下是对这个类的构造函数和前向传播函数的详细解释。

### 类的目的和作用

`TransformerEmbedding` 类的目的是为变换器模型提供一个嵌入层，它能够将词转换为词嵌入向量，并添加位置信息。这对于变换器模型来说至关重要，因为模型本身不包含处理序列顺序的机制，位置编码提供了这种能力。

### 构造函数 `__init__`

#### 参数解释

1. **vocab_size (词汇表大小)**: 这是词汇表中词的数量，决定了词嵌入矩阵的大小。

2. **d_model (模型维度)**: 表示每个词嵌入和位置编码的向量维度。

3. **max_len (最大序列长度)**: 定义了可以处理的最大序列长度，用于位置编码。

4. **drop_prob (丢弃概率)**: 在dropout层中使用，用于正则化以防止过拟合。

5. **device**: 指定计算和存储位置编码的硬件设备（如CPU或GPU）。

#### 实现细节

- `self.tok_emb`: 初始化一个 `TokenEmbedding` 类的实例，用于将词索引转换为词嵌入向量。

- `self.pos_emb`: 初始化一个 `PositionalEncoding` 类的实例，用于生成位置编码。

- `self.drop_out`: 创建一个 `nn.Dropout` 层，用于在嵌入后的向量上应用dropout。

这些组件共同工作，以生成包含词义和位置信息的嵌入向量。

### 前向传播函数 `forward`

#### 参数解释

- **x**: 输入数据，通常是一个包含词索引的整数张量。

#### 实现细节

- `tok_emb = self.tok_emb(x)`: 使用 `TokenEmbedding` 模块将输入的词索引转换为词嵌入向量。

- `pos_emb = self.pos_emb(x)`: 使用 `PositionalEncoding` 模块生成与输入相同长度的位置编码。

- `return self.drop_out(tok_emb + pos_emb)`: 将词嵌入和位置编码相加，然后应用dropout，最后返回结果。

### 与其他代码部分的联系

`TransformerEmbedding` 类在变换器模型中起着至关重要的作用。它是模型的第一层，负责将输入文本（通常是一系列词索引）转换为更丰富的表示形式，这些表示形式随后被用于模型的编码器和解码器层。通过结合词嵌入和位置编码，`TransformerEmbedding` 确保模型不仅能理解词之间的关系，而且能理解它们在文本中的相对位置。

总结来说，`TransformerEmbedding` 类为变换器模型提供了一个有效的方式来处理文本数据，它通过结合词义和位置信息，生成了一个全面的输入表示，这对于模型的性能至关重要。


 """