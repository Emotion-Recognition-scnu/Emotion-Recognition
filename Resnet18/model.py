import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #残差块
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



'''

当然，我们可以更深入地分析这段代码的方法、函数以及整体运行流程。

### 1. BasicBlock 类的详细分析
`BasicBlock` 类是构建 ResNet 的基础。它有以下几个关键部分：

#### 构造函数 `__init__`
- **参数**:
  - `in_channels`: 输入特征图的通道数。
  - `out_channels`: 输出特征图的通道数。
  - `stride`: 卷积的步长，默认为1。

- **残差函数 `self.residual_function`**:
  - 这是一个 `nn.Sequential` 容器，包含了两个 `nn.Conv2d` 卷积层和中间的 `nn.BatchNorm2d` 和 `nn.ReLU` 层。
  - 第一个卷积层调整了步长和填充，以匹配输入输出的尺寸。
  - 第二个卷积层使用了类变量 `expansion` 来保持输出通道数不变。

- **快捷路径 `self.shortcut`**:
  - 如果步长不为1或输入输出通道数不同，快捷路径通过一个额外的卷积层和批量归一化层来调整输入特征图的尺寸，使其与主路径的输出相匹配。

#### 前向传播函数 `forward`
- 输入一个特征图 `x`。
- 将 `x` 通过残差函数和快捷路径。
- 将两个路径的输出相加，并通过一个ReLU激活函数。

### 2. BottleNeck 类的详细分析
`BottleNeck` 类是另一种类型的基本块，用于更深的ResNet版本，如ResNet-50、ResNet-101等。

#### 构造函数 `__init__`
- 类似于 `BasicBlock`，但包含三个卷积层，用于更深入地提取特征。
- 第一个和最后一个卷积层用于改变通道数，中间的卷积层用于提取特征。

#### 前向传播函数 `forward`
- 功能与 `BasicBlock` 的 `forward` 方法类似。

### 3. ResNet 类的详细分析
`ResNet` 类构建了整个网络。

#### 构造函数 `__init__`
- **参数**:
  - `block`: 用于构建层的基本块类型（`BasicBlock` 或 `BottleNeck`）。
  - `num_block`: 每个卷积层中基本块的数量。

- **初始卷积层 `self.conv1`**:
  - 用于处理原始输入图像。

- **构建层 `self.conv2_x` 到 `self.conv5_x`**:
  - 使用 `_make_layer` 方法创建，每个层包含多个基本块。
  - 这些层逐渐增加特

征图的通道数，同时通过步长和池化减小其空间维度。

#### _make_layer 方法
- 用于创建一个包含指定数量基本块的层。
- **参数**:
  - `block`: 基本块类型（`BasicBlock` 或 `BottleNeck`）。
  - `out_channels`: 输出通道数。
  - `num_blocks`: 层中基本块的数量。
  - `stride`: 第一个基本块的步长。

- **过程**:
  - 根据 `num_blocks` 和 `stride` 创建基本块序列。
  - 首个基本块可能会改变特征图的大小和通道数。
  - 接下来的基本块保持特征图大小不变。

#### 前向传播函数 `forward`
- **流程**:
  - 输入数据首先通过初始卷积层 `self.conv1`。
  - 然后依次通过 `self.conv2_x`、`self.conv3_x`、`self.conv4_x` 和 `self.conv5_x`。
  - 通过自适应平均池化层 `self.avg_pool`。
  - 最后通过一个全连接层 `self.fc`，用于分类或回归任务。

### 4. ResNet18 函数
- 创建一个使用 `BasicBlock` 的 ResNet 模型，每个卷积层有两个基本块。
- 这是 ResNet-18 模型的标准配置。

### 整体运行流程
1. **初始化**: 使用 `ResNet18()` 函数初始化一个 ResNet-18 模型。
2. **输入处理**: 输入图像通过初始卷积层，进行初步的特征提取。
3. **深层特征提取**: 通过四个卷积层序列，逐渐提取更复杂的特征。在这个过程中，特征图的空间尺寸减小，而通道数增加。
4. **分类/回归**: 经过平均池化层和全连接层，输出最终的分类或回归结果。

这个代码的关键在于它使用残差连接来避免深层网络训练中的梯度消失问题，使得可以训练更深的网络以提取更丰富的特征。

现在，您对代码的哪个部分还有疑问，或者需要进一步的解释？

'''