# MNIST 数字8与9二分类数据集

从完整MNIST数据集中提取的简化版本，仅包含数字8和9。

## 数据集信息

- **类别**: 数字8和9
- **图像尺寸**: 28×28 灰度图
- **训练集**: 11,800 张 (8: 5,851张, 9: 5,949张)
- **测试集**: 1,983 张 (8: 974张, 9: 1,009张)

## 文件说明

```
processed_8_9/
├── README.md       # 本文件
├── training.pt     # 训练集
└── test.pt         # 测试集
```

## 数据加载

```python
import torch

# 加载数据
train_data = torch.load('processed_8_9/training.pt')
test_data = torch.load('processed_8_9/test.pt')

X_train, y_train = train_data  # 图像和标签
X_test, y_test = test_data

print(f"训练集: {X_train.shape}, {y_train.shape}")  # (11800, 28, 28), (11800,)
print(f"测试集: {X_test.shape}, {y_test.shape}")    # (1983, 28, 28), (1983,)
```

## 数据格式

- **图像**: `torch.Tensor`, shape `(N, 28, 28)`, dtype `uint8`, 值域 `[0, 255]`
- **标签**: `torch.Tensor`, shape `(N,)`, dtype `int64`, 值为 `8` 或 `9`

## 注意

标签保持原值(8和9)，未重新映射为0和1。
