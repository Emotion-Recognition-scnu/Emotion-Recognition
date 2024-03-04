"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
#注释
""" 

这段代码定义了一个名为MultiHeadAttention的类，它是PyTorch的nn.Module的子类，用于实现Transformer模型中的多头注意力机制。以下是其主要方法的详细步骤：

**前向传播函数 forward**：这个函数是模型的主要执行部分，它接收四个参数，q（查询），k（键），v（值）和mask（可选的掩码）。

步骤1：使用权重矩阵self.w_q，self.w_k，self.w_v对q，k，v进行线性变换。这些权重矩阵是在类的初始化过程中定义的，它们是模型的参数，会在训练过程中被优化。

步骤2：调用self.split方法将每个变换后的张量分割成self.n_head个头。这个方法会将输入张量在最后一个维度上分割成self.n_head个头，每个头的维度是d_model // self.n_head。

步骤3：调用self.attention方法计算注意力得分，并得到输出和注意力矩阵。这个方法通常会计算查询和键的点积，然后通过softmax函数得到注意力得分，最后用注意力得分对值进行加权求和。如果提供了掩码，那么在计算注意力得分之前，会将掩码加到点积的结果上。

步骤4：调用self.concat方法将输出张量的头合并回来，然后使用权重矩阵self.w_concat进行线性变换。这个方法是self.split方法的逆操作，它将输入张量的头合并回来，得到一个形状为[batch_size, length, d_model]的张量。

步骤5：这一步是可选的，用于可视化注意力矩阵。在这段代码中，这一步还没有实现。

split方法：这个方法接收一个参数，tensor（一个张量）。它将输入张量在最后一个维度上分割成self.n_head个头，每个头的维度是d_model // self.n_head。这是通过调整张量的形状并进行转置来实现的。

concat方法：这个方法接收一个参数，tensor（一个张量）。它是split方法的逆操作，它将输入张量的头合并回来，得到一个形状为[batch_size, length, d_model]的张量。这是通过转置和调整张量的形状来实现的。

这个多头注意力类的主要作用是对输入数据进行多头注意力计算。在Transformer模型中，多头注意力机制可以帮助模型同时关注输入序列的不同位置，从而更好地捕捉序列中的模式。每个头都会学习到不同的注意力模式，这可以增强模型的表达能力。

 """