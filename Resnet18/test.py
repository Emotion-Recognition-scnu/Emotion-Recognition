from model import ResNet18
import torch
from dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

batch_size = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18()
model.cuda()
model_dict = torch.load('./model_dict/ResNet.pth', map_location=device)
model.load_state_dict(model_dict['ResNet'])

dataset_test = MyDataset('./processed/test/', './processed/label.csv')
num_test = len(dataset_test)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True,
                         drop_last=True)

rmse, mae = 0., 0.
step = 0
paths, labels, predicts = [], [], []
with torch.no_grad():
    loader = tqdm(test_loader)
    for img, label, path in loader:
        paths += list(path)
        labels += torch.flatten(label).tolist()
        img, label = img.to(device), label.to(device).to(torch.float32)
        predict = model(img)
        predicts += torch.flatten(predict).tolist()
        rmse += torch.sqrt(torch.pow(torch.abs(predict - label), 2).mean()).item()
        mae += torch.abs(predict - label).mean().item()
        step += 1
        loader.set_description('step:{} {}/{}'.format(step, step * batch_size, num_test))
    rmse /= step
    mae /= step
print('Test\tMAE:{}\t RMSE:{}'.format(mae, rmse))
pd.DataFrame({'file': paths, 'label': labels, 'predict': predicts}).to_csv('testInfo.csv', index=False)



'''

这段代码是一个用于在测试集上评估深度学习模型的脚本，使用了PyTorch框架。它加载了一个预训练的模型（在这个例子中是ResNet18），然后使用这个模型在测试数据上进行预测，并计算两个重要的性能指标：均方根误差（RMSE）和平均绝对误差（MAE）。最后，它将测试结果保存到一个CSV文件中。我们将详细分析这段代码的每个部分。

### 代码分析
#### 1. 初始化和模型加载
- 设置批次大小 `batch_size` 和设备 `device`。
- 实例化 `ResNet18` 模型并将其移到GPU（如果可用）。
- 加载预训练的模型权重。

#### 2. 准备测试数据
- 使用 `MyDataset` 类（之前分析过的自定义数据集类）加载测试数据。
- 使用 `DataLoader` 创建一个数据加载器，用于批量加载测试数据，设定批量大小、是否打乱数据、是否使用内存钉扎（提高数据加载效率）和是否在数据不足一个批量时丢弃数据。

#### 3. 测试过程
- 初始化用于计算性能指标的变量：RMSE和MAE。
- 初始化用于记录测试结果的列表：`paths`（文件路径）、`labels`（真实标签）和`predicts`（预测结果）。
- 使用 `torch.no_grad()` 确保在预测过程中不计算梯度，以节省内存和计算资源。
- 遍历测试数据加载器，对每个批次的数据进行预测。
  - 将图像和标签移动到指定的设备。
  - 使用模型进行预测。
  - 计算并累加RMSE和MAE。
  - 保存路径、标签和预测结果。
  - 更新进度条信息。
- 计算整个测试集的平均RMSE和MAE。

#### 4. 结果输出和保存
- 打印测试集上的平均MAE和RMSE。
- 将文件路径、真实标签和预测结果保存到CSV文件中。

### 关键点和注意事项
- **性能评估指标**: RMSE和MAE是衡量回归任务性能的常用指标，它们提供了预测误差的量度。
- **无梯度预测**: 使用 `torch.no_grad()` 在预测时禁用梯度计算，这是评估模型时的标准做法，因为梯度在这个阶段不需要，且可以减少内存使用和加速计算。
- **数据处理**: 数据通过自定义的 `MyDataset` 类进行加载和预处理，确保数据格式与模型输入一致。
- **结果记录**: 通过将每个批次的结果累加，然后在所有批次完成后计算平均值，可以得到整个测试集的性能评估。
- **结果保存**: 测试结果被保存在CSV文件中，便于后续分析和比较。

这个脚本展示了如何在实际应用中使用训练好的深度学习模型进行预测和性能评估，同时也提供了一种方法来记录和分析预测结果。

您对这段代码还有哪些疑问，或者需要更多的解释吗？我可以进一步帮助解释代码的特定部分或其在机器学习项目中的作用。

'''