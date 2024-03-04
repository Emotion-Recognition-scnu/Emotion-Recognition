"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

#__init__函数注释
'''

""" 这段代码定义了一个名为`DecoderLayer`的类的构造函数`__init__`。这个类很可能是一个神经网络模型的一部分，特别是用于序列处理任务如机器翻译或文本生成的变换器模型中的解码器层。下面我将详细解释每个参数的作用及其在代码中的应用：

1. **d_model (模型维度)**: `d_model`指的是模型中的特征维数。在变换器模型中，这通常是词嵌入的维度。这个参数被用于设置多头注意力层（`MultiHeadAttention`）、层归一化（`LayerNorm`）和位置前馈网络（`PositionwiseFeedForward`）的输入和输出维度。确保这些子模块处理的数据维度一致对于模型的有效学习和稳定运行至关重要。

2. **ffn_hidden (前馈神经网络隐藏层大小)**: `ffn_hidden`代表前馈网络的隐藏层大小。前馈网络是解码器中的一个重要组成部分，用于在每个注意力模块之后进一步处理数据。这个参数定义了前馈网络内部的维度，通常比`d_model`大，这样可以在网络中引入更多的参数，有助于捕捉更复杂的特征。

3. **n_head (注意力头数量)**: `n_head`代表在多头注意力（`MultiHeadAttention`）机制中的头数。在变换器模型中，多头注意力允许模型在不同的子空间中并行学习信息。每个“头”都可以被看作是一个独立的注意力机制，关注输入数据的不同部分。这个参数是多头注意力机制的核心，它使得模型能够同时考虑到来自多个视角的信息。

4. **drop_prob (丢弃概率)**: `drop_prob`定义了在dropout层中使用的丢弃概率。Dropout是一种正则化技术，用于防止神经网络过拟合。通过随机丢弃一部分神经元的输出，它迫使网络学习更加鲁棒的特征表示。在这段代码中，dropout被应用于自注意力、编码器-解码器注意力和前馈网络之后的每个阶段。

构造函数中的代码首先通过调用`super(DecoderLayer, self).__init__()`初始化父类。然后，它使用上述参数构建了几个关键的网络层：

- `self.self_attention`和`self.enc_dec_attention`都是`MultiHeadAttention`层，分别用于自注意力和编码器-解码器注意力。
- `self.norm1`、`self.norm2`和`self.norm3`是`LayerNorm`层，用于归一化上一个层的输出，帮助稳定训练过程。
- `self.dropout1`、`self.dropout2`和`self.dropout3`是`nn.Dropout`层，用于应用dropout正则化。
- `self.ffn`是`PositionwiseFeedForward`层，它提供了一个额外的、非线性的处理步骤，帮助网络学习复杂的特征表示。

总之，这些参数共同定义了解码器层的结构和行为，使其能够有效地处理和生成序列数据。 """

'''



    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


#forward函数注释
""" '''
这个`forward`函数是一个神经网络（特别是变换器模型的解码器部分）的核心组成部分。它的作用是对给定的输入执行一系列操作以生成输出，这些操作包括注意力机制、层归一化和前馈网络。下面我将详细解释每个部分的作用和实现方式。

### 函数作用和准备目的

`forward`函数是用于神经网络的前向传播过程，即给定输入数据后，如何计算并输出结果的过程。它在序列到序列的模型（如机器翻译、文本摘要等）中尤为关键，用于将编码器的输出（对输入序列的表示）转换为最终的输出序列。

### 参数解释

1. **dec (解码器输入)**: 这是当前解码器层的输入。在一个序列到序列的任务中，它通常是前一个解码器层的输出或者初始的目标序列表示。

2. **enc (编码器输出)**: 这是编码器层的输出，包含了输入序列的编码信息。在编码器-解码器注意力机制中，这个输出被用作键（key）和值（value）。

3. **trg_mask (目标掩码)**: 这是一个掩码（mask），用于在自注意力计算中避免未来信息的泄露。在解码器中，由于输出是逐步生成的，因此需要确保在生成第`n`个词时不会看到第`n+1`个词的信息。

4. **src_mask (源掩码)**: 这是另一个掩码，用于在编码器-解码器注意力中屏蔽掉无关的输入部分，比如输入序列中的填充（padding）部分。

### 函数实现细节

1. **自注意力计算**: 首先，函数对解码器的输入`dec`执行自注意力计算。自注意力允许解码器中的每个位置都能够关注到解码器输入的所有位置，这对于捕捉序列内的依赖关系很重要。

2. **添加和归一化**: 接着，使用dropout进行正则化，然后将自注意力的输出与原始输入`dec`相加，最后通过层归一化（`LayerNorm`）进行归一化处理。这个“添加和归一化”的步骤有助于避免梯度消失问题，同时保持网络的深度。

3. **编码器-解码器注意力**: 如果提供了编码器的输出`enc`，函数将执行编码器-解码器注意力计算。这一步骤使得解码器能够关注到编码器输出的相关部分，是序列到序列模型的关键环节。

4. **位置前馈网络**: 最后，通过位置前馈网络（`PositionwiseFeedForward`）进行处理。这是一个全连接的前馈网络，可以为模型提供额外的非线性处理能力。

5. **再次添加和归一化**: 和之前一样，每次处理后都会进行一次“添加和归一化”的操作，以保持网络的稳定性和有效性。

### 与其他代码块的联系

这个`forward`函数是变换器模型中解码器层的核心。它通常被嵌入在更大的模型结构中，与编码器层以及可能的其他解码器层相互作用。在整个模型的训练和推断过程中，`forward`函数负责处理和传递信息，是生成最终输出的关键步骤。
''' """