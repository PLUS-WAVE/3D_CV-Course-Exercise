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

## Recognition

使用经典图像处理的方法开发简单人物检测器，其大致步骤如下：

1. 获取正样本训练数据，即包含人的图像块。获取负样本训练数据，即不包含（完整）人的图像块。
2. 提取方向梯度直方图（HoG）特征，以获取比原始像素值**更稳健**的图像描述符。
3. 使用这些训练数据训练我们选择的分类器。使用一个简单的最近邻搜索或支持向量机。
4. 使用滑动窗口方法从我们的验证图像中提取图像块和HoG特征，并在每个图像块上评估分类器，以便在验证图像中检测出人。

### 1 获取训练图像块

通过已经有的图像及数据，我们可以获得正样本训练数据，即包含人的图像块，使用PIL库进行裁剪

```python
assert isinstance(img, Image.Image), 'img needs to be PIL.Image.Image'
crop = img.crop(box)  
patch = crop.resize(patch_size)
return patch
```

再通过将给定大小的框放置在图像中的随机位置，获得负样本训练数据；

```python
x = np.random.randint(0, W - boxsize[0])
y = np.random.randint(0, H - boxsize[1])
box = [x, y, x + boxsize[0], y + boxsize[1]]
```

为了包括更具挑战性的负样本，我们需要在已有的边界框基础上添加一个**小**的随机偏移量，从而构建出与正样本训练数据相似的困难的负样本。

```python
off_x = np.random.randint(min_offset, max(2, max_offset_w))
off_y = np.random.randint(min_offset, max(2, max_offset_h))
off_box = [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y]
```

<img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-05-24/image-20240524091815978.png" alt="image-20240524091815978" style="zoom: 80%;" />

### 2 提取 HoG 特征

为了构建一个更鲁棒的人体检测器，我们将使用方向梯度直方图（Histogram of Oriented Gradients，HoG）特征。HoG 特征在处理视角、光照变化以及小的形变（如平移、缩放、旋转、透视变换）时具有较强的鲁棒性。

##### 基本思想：

HoG 的核心思想是通过梯度方向和梯度幅值来表示图像块。具体，我们将图像块的每个像素点的梯度角度（由梯度方向决定）和梯度幅值（由梯度大小决定）用直方图表示，从而得到该图像块的特征描述。

##### 提取步骤：

1. **梯度计算**

   - **梯度幅值 (Gradient Magnitude)**：首先，对输入图像进行梯度计算，得到每个像素点的梯度幅值（强度）和梯度方向。梯度幅值描述了像素值变化的程度，梯度方向描述了变化的方向。

   - **梯度方向 (Gradient Angle)**：计算每个像素点的梯度方向，并将这些方向量化为若干个方向区间。例如，这里将方向区间划分为8个方向（每个方向45度），这样可以简化计算和后续处理。

2. **图像划分**

   将图像划分为多个较小的单元（cells），每个单元包含8x8个像素。

3. **构建方向直方图**

   方向量化和加权：将梯度方向映射到相应的方向区间，并将梯度幅值作为该区间的权重值加进去。

4. **块归一化**

   将若干个单元组合成一个块，例如2x2的单元构成一个块。然后，对块内的梯度方向直方图进行归一化处理，以增强特征的鲁棒性。

5. **特征向量构建**

   将所有块的归一化直方图拼接在一起，形成最终的HoG特征向量。

<img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-05-24/image-20240524092313289.png" alt="image-20240524092313289" style="zoom: 67%;" />

我们可以使用 scikit-image 库来计算图像的HoG特征。

```python
hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), channel_axis=-1, **kwargs)
```

- **img**：输入图像，类型可以是2D灰度图或3D彩色图
- **orientations**：方向区间的数量，即将梯度方向划分为多少个区间。默认值为8
- **pixels_per_cell**：每个单元（cell）包含的像素数，通常为一个2元组，例如(16, 16)，表示每个单元为16x16个像素
- **cells_per_block**：每个块（block）包含的单元数，通常为一个2元组，例如(1, 1)，表示每个块包含1x1个单元，即不进行**块归一化**
- **channel_axis**：图像通道所在的轴，对于彩色图像，通常为-1表示最后一个轴

### 3 训练分类器

将之前获取的正和负训练图像块进行HoG特征提取后拼接在一起，得到训练使用的**特征向量**`X`

```python
for p in positives:
    fds_pos.append(get_hog(p))
for n in negatives:
    fds_neg.append(get_hog(n))
X = np.stack(fds_pos + fds_neg)
```

再使用正样本（标签是1）创建了一个全为1的布尔数组，长度等于正样本的数量；负样本（标签是0）创建了一个全为0的布尔数组，长度等于负样本的数量。这两个数组被连接在一起，形成了**目标向量**`y`。

```python
y = np.concatenate([np.ones(len(positives), dtype=np.bool), np.zeros(len(negatives), dtype=np.bool)])
```

##### 最近邻分类器：

使用Faiss库：

```python
d = X.shape[1]
quantizer = faiss.IndexFlatL2(d) # measure L2 distance
index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2) # build the index

index.train(X.astype(np.float32))
index.add(X.astype(np.float32)) # add vectors to the index
```

##### 支持向量机SVM：

使用 `sklearn.svm.SVC` 训练支持向量机。

```python
svm = SVC(class_weight='balanced') # use balanced weight since we have more negatives than positives
svm.fit(X, y)
```

### 4 Evaluate

为了评估分类器，我们需要从目标图像中提取图像块及其 HOG 特征，以便我们的目标数据与训练数据进行对比。我们使用滑动窗口方法：

1. **在图像上滑动窗口**：

   - 从图像的左上角开始，将窗口放置在该位置。提取窗口覆盖的图像区域作为一个裁剪。
   - 根据步长在图像上移动窗口，提取新的裁剪。
   - 重复上述步骤，直到窗口覆盖完整个图像。

   ```python
   for y in range(0, H - window_size[1], step_size):
       for x in range(0, W - window_size[0], step_size):
           window = (x, y, x + window_size[0], y + window_size[1])
   ```

2. **特征提取与分类**：

   - 对每个裁剪提取HoG特征。
   - 使用预先训练好的分类器对每个裁剪进行分类，判断其是否包含目标。