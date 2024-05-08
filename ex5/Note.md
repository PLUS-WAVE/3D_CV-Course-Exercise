## Occupancy Network

OccNet 的关键思想是**隐式地表示3D形状**，而不是显式地表示。与直接编码形状几何信息不同，OccNet 将形状的表面建模为非线性分类器的**决策边界**。

- **隐式表示**：Occupancy Networks 将 3D 形状表示为非线性分类器函数的决策边界
   $$
   f_{\theta} : \mathbb{R}^2 \times X \rightarrow [0,1]
   $$

   这里，$X$​ 表示输入空间（例如，体素网格或点云），函数在给定点的输出表示该点的是否占用（**该点在物体内部还是外部**，下图中即红色在内部，蓝色在外部）

   <img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-05-08/image-20240508100153121.png" alt="image-20240508100153121" style="zoom:67%;" />

### 1 数据集准备

首先我们导入训练数据，即我们想要表示的 3D 对象，其包含 100k 个采样点及其占用值。**占用值**指示该点是属于对象 ( `occupancy=1` ) 还是不属于 ( `occupancy=0` )

<img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-05-08/image-20240508100435255.png" alt="image-20240508100435255" style="zoom:50%;" />

我们首先需要将数据分为训练集和验证集。随机分割数据，使用 80% 作为训练集，20% 作为验证集。

### 2 创建神经网络

这个神经网络即为一个分类器，其传入的3维的坐标，输出1维的占用值0/1。简单设计为具有 4 个隐藏层，隐藏维度为 64 的网络（总共 6 层，4 个隐藏层 + 1 个输入 + 1 个输出）

```python
class OccNet(nn.Module):
    def __init__(self, size_h=64, n_hidden=4):
        super().__init__()
        
        # 4 个隐藏层，隐藏维度为 64（总共 6 层，4 个隐藏层 + 1 个输入 + 1 个输出）
        layers = [nn.Linear(3, size_h), nn.ReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(size_h, size_h), nn.ReLU()]
        layers += [nn.Linear(size_h, 1)]
    
        self.main = nn.Sequential(*layers)
    
    def forward(self, pts):
        return self.main(pts).squeeze(-1)       # squeeze dimension of the single output value
```

然后我们定义损失函数、优化器：

1. **损失函数 (Loss Function)**:

   `nn.BCEWithLogitsLoss`，二元交叉熵损失和 softmax 函数的结合。这个损失函数通常用于二分类问题；参数`reduction='none'` 使损失在计算时不进行求和或平均，而是对每个样本点都会产生一个独立的损失值

2. **优化器 (Optimizer)**:

   - 使用了 Adam 优化器，它是一种常用的随机梯度下降优化算法的变体。Adam 优化器具有自适应学习率的特性，可以在训练过程中根据梯度的不同情况来调整学习率。

```python
model = OccNet(size_h=64, n_hidden=4)
criterion = nn.BCEWithLogitsLoss(reduction='none')    # binary cross entropy loss + softmax
optimizer = torch.optim.Adam(model.parameters())
```

### 3 训练

1. 训练循环：

   对于每个批次的采样点和占用值，将数据送入模型进行**前向传播**，计算输出和损失使用优化器**清零梯度**，并进行**反向传播**和**权重更新**

   ```python
   optimizer.zero_grad()  # 清零梯度
   output = model(pts)  # 前向传播，通过模型计算预测值
   loss = criterion(output, occ).mean()  # 计算损失
   loss.backward()  # 反向传播，计算每个参数的梯度
   optimizer.step()  # 更新权重，使用梯度进行一步优化算法
   ```

2. 验证：

   - 在每个 epoch 或每个指定的迭代次数后，使用验证集计算模型的验证损失
   - 如果验证损失比之前记录的最佳验证损失还要小，则保存当前模型

正常的训练loss曲线：

<img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-05-08/image-20240508151543465.png" alt="image-20240508151543465" style="zoom:50%;" />

### 4 从隐式表示中提取对象

之前我们已经训练好了一个神经网络，现在需要使用经过训练的网络来恢复物体的 3D 形状。

1. 采样点网格：首先，在三维空间内生成一组等间距采样点网格，这些点将用于评估网络的输出

   参数 `xmin`、`xmax` 和 `resolution`，分别表示每个轴上的最小值、最大值和分辨率

   ```python
   def make_grid(xmin, xmax, resolution):
     grid_1d = torch.linspace(xmin, xmax, resolution)
     grid_3d = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='xy'), -1)
     return grid_3d.flatten(0, 2)     # RxRxRx3 -> (R^3)x3 
   ```

   例如：使用 `resolution = 128` 设置每个维度上的分辨率为 128，即得到 128×128×128 的三维网格，共  $128^3$ 个点

2. 预测占用值：将采样点输入经过训练的网络中，获取每个点处的占用值

3. 网格生成：基于占用值，使用**网格生成算法**（如 Marching Cubes）生成三维网格，该网格将近似地表示原始的 3D 形状