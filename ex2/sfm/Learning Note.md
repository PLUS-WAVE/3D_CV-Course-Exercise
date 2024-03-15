### 1 八点法计算F矩阵（基础矩阵）

**基础矩阵**用于描述两个视图之间的几何关系



1. 基础矩阵：基础矩阵 $F$ 是描述两个视图之间相机投影关系的矩阵。对于两个对应的图像坐标点 $(x, y, 1)$ 和 $(u, v, 1)$​ 在两个视图上，基础矩阵满足以下方程：

   这个方程即**对极约束**，描述了图像中对应点的投影关系

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}^T \cdot F \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = 0
$$

 

2. 线性系统：对于多对对应点，可以构建一个线性方程系统 $Af = 0$ ，其中 $A$ 是由对应点生成的矩阵， $f$​ 是基础矩阵的扁平形式

   上述方程即：
   
$$
\begin{bmatrix} u & v & 1 \end{bmatrix} \cdot \begin{bmatrix} f_{11} & f_{12} & f_{13} \\ f_{21} & f_{22} & f_{23} \\ f_{31} & f_{32} & f_{33} \end{bmatrix} \cdot \begin{bmatrix} x \\ 
y \\ 
1 \end{bmatrix} = 0
$$

​	展开得到：

$$
\begin{bmatrix} ux&vx&x&uy&vy&y&u&v&1 \end{bmatrix}\cdot \begin{bmatrix}f_{11} \\
f_{12} \\ 
f_{13} \\ 
f_{21} \\ 
f_{22} \\ 
f_{23} \\ 
f_{31} \\ 
f_{32} \\
f_{33} \\ \end{bmatrix} = 0
$$

​	这个矩阵方程可以表示为 $A_if = 0$​ 

​	为了解出这个9个未知数的 $f$ ，我们至少需要**8对点**，所以叠加 $A_i$ 得到 $A$ 矩阵

$$
A = \begin{bmatrix} x_1u_1 & x_1v_1 & x_1 & y_1u_1 & y_1v_1 & y_1 & u_1 & v_1 & 1 \\
x_2u_2 & x_2v_2 & x_2 & y_2u_2 & y_2v_2 & y_2 & u_2 & v_2 & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_8u_8 & x_8v_8 & x_8 & y_8u_8 & y_8v_8 & y_8 & u_8 & v_8 & 1 \end{bmatrix}
$$


3. 最小二乘法：通过奇异值分解（SVD），取 $V^T$ 的最后一列作为估计矩阵 $A$ 的最小二乘解，即 $f$ 

   > <u>方程的最小二乘解有一个既定的结论，即对 $A$ 进行SVD分解</u>，得到的 == $V^T$ 的最后一行== 即是 $f$ 的解

4. 基础矩阵还原：将 $f$ reshape 为 $3 \times 3$​ 的矩阵，然后通过奇异值分解（SVD）对矩阵进行调整，以确保基础矩阵的秩为2

   - SVD分解：
      对矩阵 $F$ 进行奇异值分解：$F = U \Sigma V^T$ ，其中 $U$ 和 $V$ 是正交矩阵，$\Sigma$ 是对角矩阵

   - 秩-2约束：
      将奇异值矩阵 $\Sigma$ 调整为仅保留前两个奇异值（**将第三个奇异值设为0**），以<u>确保基础矩阵的秩为2</u>

   - 重构基础矩阵：
       $F = U \Sigma' V^T$

   ```python
   F = f.reshape((3, 3))
   
   # 对F进行SVD分解
   U, S, Vt = np.linalg.svd(F)
   
   # 将奇异值矩阵Sigma调整为仅保留前两个奇异值（第三个设为0）
   S[2] = 0
   
   # 重构基础矩阵F
   F = np.dot(U, np.dot(np.diag(S), Vt))
   ```

5. 归一化：对基础矩阵进行归一化，以确保尺度的一致性

### 2 标准化八点算法

对普通的八点算法进行了改进，通过标准化输入数据，提高了算法的稳健性和准确性

1. 我们首先将对应点标准化为零均值和单位方差，以消除尺度的影响
   ```python
   mean1 = np.mean(keypoints1, axis=0)
   mean2 = np.mean(keypoints2, axis=0)
   std1 = np.std(keypoints1, axis=0)
   std2 = np.std(keypoints2, axis=0)
   # 防止除0，由于齐次坐标，标准差std算得最后一项为0
   std1[2] = 1
   std2[2] = 1
   nomalized_points1 = (keypoints1 - mean1) / std1
   nomalized_points2 = (keypoints2 - mean2) / std2
   ```
	
$$
\bar{x} = \frac{x - \bar{\mu_x}}{\sigma_x}
$$



   也等于左乘一个转换矩阵 $T$ ：


$$
T = \begin{bmatrix} \frac{1}{\sigma_x} & 0 & -\frac{\mu_x}{\sigma_x} \\
0 & \frac{1}{\sigma_y} & -\frac{\mu_y}{\sigma_y} \\ 
0 & 0 & 1 \end{bmatrix}
$$


2. 在这些标准化点上运行八点算法

3. 最后对得到的基本矩阵进行反变换，在计算基础矩阵后，需要将其进行撤销标准化处理，以获得最终的基础矩阵

$$
F = T_2^{-1} \cdot F_{normalized} \cdot T_1
$$

   



### 3 三角测量

我们有两个相机，它们的c分别为 $P_1$ 和 $P_2$ （ $3 \times 4$​ 矩阵）。

$$
P = K\begin{bmatrix}R|t\end{bmatrix}
$$

对于一个在相机1和相机2中分别观察到的同一物体的对应点 $\tilde x_1$ 和 $\tilde x_2$ （齐次坐标 $3 \times 1$ 向量） ，我们可以得到以下方程：其中，$\tilde X$ （齐次坐标 $4 \times 1$ 向量）是物体在三维空间中的坐标

$$
P_1 \tilde X =\tilde x_1\\
P_2 \tilde X =\tilde x_2
$$

 将 $P$ 分解为三个向量：

$$
\displaylines{
P_i =\begin{bmatrix}P_{i1}\\ 
P_{i2} \\
P_{i3}
\end{bmatrix} \\
P_{i1} = [p_{11}, p_{12}, p_{13}, p_{14}] \\
P_{i2} = [p_{21}, p_{22}, p_{23}, p_{24}] \\
P_{i3} = [p_{31}, p_{32}, p_{33}, p_{34}] \\
}
$$

这样，原等式就变为：

$$
\begin{bmatrix}P_{i1}\tilde X \\
P_{i2}\tilde X \\
P_{i3}\tilde X\end{bmatrix}
=\begin{bmatrix}x_i \\
y_i \\
1\end{bmatrix}
$$

将左边向量齐次化除以第三个元素，与右边向量元素一一对应：

$$
\displaylines{
P_i \tilde X = \begin{bmatrix} \frac{P_{i1} \tilde X}{P_{i3}\tilde X} \\ 
\frac{P_{i2} \tilde X}{P_{i3} \tilde X} \\
1 \end{bmatrix}= \begin{bmatrix}x_i \\
y_i \\
1 \end{bmatrix}
= \tilde x_i \\
x_i = \frac{P_{i1} \tilde X}{P_{i3} \tilde X} \Rightarrow x_iP_{i3} \tilde X-P_{i1} \tilde X = 0 \\
y_i = \frac{P_{i2} \tilde X}{P_{i3} \tilde X} \Rightarrow y_iP_{i3} \tilde X-P_{i2} \tilde X = 0
}
$$

由于我们知道 $x_1$ 、 $x_2$ 和 $P_1$ 、 $P_2$​​ ，我们可以将其转化为一个**齐次线性方程组**：

$$
\displaylines{
A_1 = \begin{bmatrix} x_1 P_{13} - P_{11} \\ 
	y_1 P_{13} - P_{12} \end{bmatrix} 
\\
A_2 = \begin{bmatrix} x_2 P_{23} - P_{21} \\ 
	y_2 P_{23} - P_{22} \end{bmatrix}
\\
A = \begin{bmatrix}A_1 \\
	A_2 \end{bmatrix} 
\\ 
A\tilde X = 0
}
$$

```python
A = np.array(
    [keypoint1[0] * P1[2] - P1[0],
     keypoint1[1] * P1[2] - P1[1],
     keypoint2[0] * P2[2] - P2[0],
     keypoint2[1] * P2[2] - P2[1]]
)
```

这样我们就可以使用**最小二乘法**或其他方法来解决这个线性方程组，从而找到物体的三维位置 $X$​ 

```python
# DLT算法解决最小二乘法
_, _, Vt = np.linalg.svd(A)
x_w = Vt[-1]
x_w = x_w / x_w[3] # 齐次坐标
```

