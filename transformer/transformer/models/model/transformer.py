"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    

#注释
""" 

这个`Transformer`类是一个完整的Transformer模型的实现，它包含了一个编码器（`Encoder`）和一个解码器（`Decoder`）。这个类的主要作用是将编码器和解码器连接起来，以便可以一次性地对整个模型进行训练和推理。

1. **`__init__`方法**：这个方法是类的构造函数，用于初始化类的实例。它接收一系列参数，包括源语言和目标语言的填充索引（`src_pad_idx`和`trg_pad_idx`），目标语言的开始符号索引（`trg_sos_idx`），编码器和解码器的词汇表大小（`enc_voc_size`和`dec_voc_size`），模型的维度（`d_model`），多头注意力的头数（`n_head`），序列的最大长度（`max_len`），前馈神经网络的隐藏层大小（`ffn_hidden`），层的数量（`n_layers`），丢弃概率（`drop_prob`），以及设备（`device`）。这些参数都会被存储在类的实例中，以便在后面的计算中使用。

   在这个方法中，还会创建一个编码器和一个解码器。这两个组件都是Transformer模型的关键部分，它们分别负责将输入序列转换为中间表示，以及将中间表示转换为输出序列。

2. **`forward`方法**：这个方法是模型的前向传播函数，用于计算模型的输出。它接收两个参数，`src`（源语言序列）和`trg`（目标语言序列）。

   - **步骤1**：首先，使用`make_src_mask`和`make_trg_mask`方法为源语言序列和目标语言序列创建掩码。这些掩码用于在计算注意力得分时忽略填充位置。

   - **步骤2**：然后，将源语言序列和源语言掩码传递给编码器，得到编码后的源语言序列。

   - **步骤3**：最后，将目标语言序列、编码后的源语言序列、目标语言掩码和源语言掩码传递给解码器，得到模型的输出。

3. **`make_src_mask`和`make_trg_mask`方法**：这两个方法用于创建源语言序列和目标语言序列的掩码。它们的工作原理是，对于每个位置，如果该位置的词是填充词，那么掩码的对应位置就是`False`，否则就是`True`。这些掩码会被用在注意力机制中，以便在计算注意力得分时忽略填充位置。

这个类的实现依赖于`Encoder`和`Decoder`两个类。这两个类分别实现了Transformer模型的编码器和解码器，它们的具体实现可能会在其他的Python文件中。在这个类中，编码器和解码器的参数是通过构造函数传递的，这样可以保证编码器和解码器的参数和整个模型的参数是一致的。

这个类的实现也依赖于PyTorch库，这是一个用于深度学习的开源库。在这个类中，PyTorch的`nn.Module`类被用作基类，这样可以利用PyTorch的自动求导和优化器等功能。此外，这个类中的张量操作（如创建掩码和计算输出）也都是使用PyTorch的函数实现的。

 """