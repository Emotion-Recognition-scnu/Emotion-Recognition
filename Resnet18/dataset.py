from torch.utils.data import Dataset
from load_data import load
import torchvision
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.path, self.label = load(img_path, label_path)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.path[idx])
        img = self.transform(img)
        return img, self.label[idx], self.path[idx]

'''

这段代码定义了一个名为 `MyDataset` 的类，它是 PyTorch 中 `Dataset` 类的子类。这个自定义数据集用于加载和预处理图像数据。让我们逐步分析这个类的结构和功能。

### MyDataset 类
`MyDataset` 类继承自 PyTorch 的 `Dataset` 类，用于处理图像数据集。

#### 构造函数 `__init__`
- **参数**:
  - `img_path`: 存储图像文件的路径。
  - `label_path`: 存储标签的路径。

- **功能**:
  - 使用 `load` 函数（可能是自定义的）从 `img_path` 和 `label_path` 加载图像路径和标签。
  - 初始化 `transform` 属性，该属性是一个 `torchvision.transforms.Compose` 对象，用于执行一系列图像变换操作。

- **图像变换 `self.transform`**:
  - `torchvision.transforms.Resize((128, 128))`: 将图像大小统一调整为 128x128 像素。
  - `torchvision.transforms.ToTensor()`: 将 PIL 图像或 NumPy `ndarray` 转换为 `torch.Tensor`。
  - `torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: 标准化图像，使用预定义的均值和标准差。

#### `__len__` 方法
- 返回数据集中的样本总数，这是通过获取 `self.label` 的长度实现的。

#### `__getitem__` 方法
- **参数**:
  - `idx`: 指定要获取的样本的索引。

- **功能**:
  - 使用索引 `idx` 从 `self.path` 中获取图像的路径。
  - 使用 `Image.open` 打开图像。
  - 应用 `self.transform` 对图像进行预处理。
  - 返回处理后的图像、对应的标签和图像路径。

### 数据加载和预处理流程
1. **初始化**: 实例化 `MyDataset` 类时，首先调用 `load` 函数加载图像路径和标签。
2. **图像变换**: 通过 `Compose` 创建一个变换序列，以便在数据加载时对每个图像执行这些变换。
3. **获取数据**: 当从数据集中请求一个项目时，`__getitem__` 方法负责加载图像，应用预定义的变换，并返回变换后的图像和相应的标签。

### 关键点和可能的注意事项
- **数据加载**: `load` 函数的具体实现未在代码中给出，但它对于如何读取和解析图像路径和标签至关重要。
- **图像预处理**: 变换序列的选择对于模型训练的效果有显著影响。这里包括了大小调整、转换为张量和标准化，这些都是深度学习中常见的图像预处理步骤。
- **数据集大小**: `__len__` 方法提供了一种获取数据集大小的方式，这对于迭代和评估数据集非常重要。

您对这段代码的哪个部分还有疑问，或者需要进一步的解释吗？我可以帮助您深入理解特定部分的实现和原理。

'''