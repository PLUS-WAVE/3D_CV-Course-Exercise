### 1 Lighting

要完成 `get_face_color` 功能，需要根据给定的法向量和点光源方向，使用渲染方程计算光强度。渲染方程为：

$$
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) (\mathbf{n} \cdot \omega_i) d\omega_i
$$

但是，在该情况下，假设表面不发光 ( $L_e(\mathbf{x}, \omega_o) = 0$ ) 并且 BRDF 项 ( $f(\mathbf{x}, \omega_i, \omega_o)$ ) 始终为 1 ，因此，简化形式变为：

$$
L_o(\mathbf{x}, \omega_o) = \int_{\Omega} L_i(\mathbf{x}, \omega_i) (\mathbf{n} \cdot \omega_i) d\omega_i
$$


对于点光源，可以进一步简化为：

$$
L_o(\mathbf{x}, \omega_o) = L_i(\mathbf{x}, \omega_i) (\mathbf{n} \cdot \omega_i)
$$


$L_i(\mathbf{x}, \omega_i)$ 是入射光强度为 `1` ， $\omega_i$ 是光方向，代码实现：

```python
light_intensity = np.dot(normal, -point_light_direction)
```

> normal 与 point_light_direction 夹角为钝角，取负数使点乘后结果为正



### 2 Projection

**透视投影** *Perspective-Project* 和**正交投影** *Orthographic-Projecte* 最相似的情况通常是当透视投影的<u>焦距非常大</u>时。在透视投影中，焦距决定了投影的“透视效应”有多强。当焦距非常大时，透视效应几乎被消除，因此透视投影的结果看起来和正交投影非常相似。



### 3 DLT算法求解单应性矩阵

#### 原理：

单应性矩阵描述了两个图像之间的投影变换关系

下面是DLT算法的基本原理：

1. **构建投影方程：** 对于两个图像中的对应点 $(x, y, 1)$ 和 $(u, v, 1)$ ，投影关系可以用齐次坐标表示为 $c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$ 。这里的 $H$（ $3 \times 3$ 矩阵）是我们要求解的单应性矩阵

$$
H =\begin{bmatrix}
h1 & h2 & h3 \\
h4 & h5 & h6 \\
h7 & h8 & h9
\end{bmatrix}
$$

2. **构建矩阵 $A$：** 将投影方程展开成 $Ah = 0$ 的形式，其中 $A$ 是一个 $2n \times 9$ 的矩阵，$h$ 是包含矩阵 $H$ 所有元素的列向量
   
$$
A = \begin{bmatrix}
-x_1 & -y_1 & -1 & 0 & 0 & 0 & u_1x_1 & u_1y_1 & u_1 \\
0 & 0 & 0 & -x_1 & -y_1 & -1 & v_1x_1 & v_1y_1 & v_1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
-x_n & -y_n & -1 & 0 & 0 & 0 & u_nx_n & u_ny_n & u_n \\
0 & 0 & 0 & -x_n & -y_n & -1 & v_nx_n & v_ny_n & v_n \\
\end{bmatrix}
$$

$$
h=\begin{bmatrix}h1&h2&h3&h4&h5&h6&h7&h8&h9\end{bmatrix}
$$

3. **奇异值分解（SVD）：** 对矩阵 $A$ 进行奇异值分解，得到 $A = U \Sigma V^T$。取 $V^T$ 的最后一列作为 $h$​ 的估计

   > <u>方程的最小二乘解有一个既定的结论，即对 $A$ 进行SVD分解</u>，得到的 == $V^T$ 的最后一行== 即是 $h$ 的解，对 $h$ 做 reshape 得到 $H$ 。



#### 实现：

根据你提供的信息，**DLT（Direct Linear Transform）算法**用于通过最小二乘法来估计单应性矩阵 $H$，以拟合两组特征点之间的关系。下面是DLT算法的具体步骤：

1. **构建矩阵 $A$：** 对于每一对特征点 $(x, y, 1)$ 和 $(u, v, 1)$，构建一个对应的矩阵 $A_i$。将所有这些矩阵堆叠成一个大矩阵 $A$ 

$$
A_i = \begin{bmatrix}
-x & -y & -1 & 0 & 0 & 0 & ux & vy & uv \\
0 & 0 & 0 & -x & -y & -1 & ux & vy & uv \\
\end{bmatrix}
$$

2. **SVD分解：** 对矩阵 $A$ 进行奇异值分解（SVD），得到 $A = U \Sigma V^T$ 。取 $V^T$ 矩阵的最后一行作为矩阵 $h$ 

3. **Reshape：** 将向量 $h$ reshape 为 $3 \times 3$ 的单应性矩阵 $H$ 

