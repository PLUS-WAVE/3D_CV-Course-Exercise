# import modules
import numpy as np
import matplotlib.pyplot as plt
import imageio

# load and plot input image
img = imageio.imread('assets/gfx/image.png') / 255
plt.imshow(img, interpolation='nearest');
plt.gray()
plt.show()

# model parameters
[h, w] = img.shape  # get width & height of image
num_vars = w * h  # number of variables = width * height
num_states = 2  # binary segmentation -> two states

# # initialize factors (list of dictionaries), each factor comprises:
# #   vars: array of variables involved
# #   vals: vector/matrix of factor values
# factors = []
#
# # 添加一元因子
# for u in range(w):
#     for v in range(h):
#         # 从图像中获取观测值
#         obs = img[v, u]
#
#         # 创建一元因子，其中 'vars' 是变量的索引，'vals' 是对应的因子值
#         unary_factor = {'vars': [v * w + u], 'vals': np.array([1 - obs, obs])}
#         factors.append(unary_factor)
#
# # 添加二元因子
# alpha_val = 0.4  # 平滑权重
# E = alpha_val * np.array([[1, 0], [0, 1]])  # 二元因子的能量矩阵
# for u in range(w):
#     for v in range(h):
#
#         # 计算当前像素的索引
#         var_index = v * w + u
#
#         # 创建二元因子，其中 'vars' 是变量的索引，'vals' 是对应的因子值
#         pairwise_factor = {'vars': [var_index, var_index + 1], 'vals': np.exp(E)}
#
#         # 如果不是最后一列，添加到因子列表中
#         if u < w - 1:
#             factors.append(pairwise_factor)
#
#         # 如果不是最后一行，添加到因子列表中
#         if v < h - 1:
#             pairwise_factor = {'vars': [var_index, var_index + w], 'vals': np.exp(E)}
#             factors.append(pairwise_factor)
#
# # initialize all messages
# msg_fv = {}  # f->v messages (dictionary)
# msg_vf = {}  # v->f messages (dictionary)
# ne_var = [[] for i in range(num_vars)]  # neighboring factors of variables (list of list)
#
# # set messages to zero; determine factors neighboring each variable
# for [f_idx, f] in enumerate(factors):
#     for v_idx in f['vars']:
#         msg_fv[(f_idx, v_idx)] = np.zeros(num_states)  # factor->variable message
#         msg_vf[(v_idx, f_idx)] = np.zeros(num_states)  # variable->factor message
#         ne_var[v_idx].append(f_idx)  # factors neighboring variable v_idx
#
# # status message
# print("Messages initialized!")
# # run inference
# for it in range(30):
#
#     # for all factor-to-variable messages do
#     for [key, msg] in msg_fv.items():
#
#         # shortcuts to variables
#         f_idx = key[0]  # factor (source)
#         v_idx = key[1]  # variable (target)
#         f_vars = factors[f_idx]['vars']  # variables connected to factor
#         f_vals = factors[f_idx]['vals']  # vector/matrix of factor values
#
#         # unary factor-to-variable message
#         if np.size(f_vars) == 1:
#             msg_fv[(f_idx, v_idx)] = f_vals
#
#         # pairwise factor-to-variable-message
#         else:
#
#             # if target variable is first variable of factor
#             if v_idx == f_vars[0]:
#                 msg_in = np.tile(msg_vf[(f_vars[1], f_idx)], (num_states, 1))
#                 msg_fv[(f_idx, v_idx)] = (f_vals + msg_in).max(1)  # max over columns
#
#             # if target variable is second variable of factor
#             else:
#                 msg_in = np.tile(msg_vf[(f_vars[0], f_idx)], (num_states, 1))
#                 msg_fv[(f_idx, v_idx)] = (f_vals + msg_in.transpose()).max(0)  # max over rows
#
#         # normalize
#         msg_fv[(f_idx, v_idx)] = msg_fv[(f_idx, v_idx)] - np.mean(msg_fv[(f_idx, v_idx)])
#
#     # for all variable-to-factor messages do
#     for [key, msg] in msg_vf.items():
#
#         # shortcuts to variables
#         v_idx = key[0]  # variable (source)
#         f_idx = key[1]  # factor (target)
#
#         # add messages from all factors send to this variable (except target factor)
#         # and send the result to the target factor
#         msg_vf[(v_idx, f_idx)] = np.zeros(num_states)
#         for f_idx2 in ne_var[v_idx]:
#             if f_idx2 != f_idx:
#                 msg_vf[(v_idx, f_idx)] += msg_fv[(f_idx2, v_idx)]
#
#         # normalize
#         msg_vf[(v_idx, f_idx)] = msg_vf[(v_idx, f_idx)] - np.mean(msg_vf[(v_idx, f_idx)])
#
# # calculate max-marginals (num_vars x num_states matrix)
# max_marginals = np.zeros([num_vars, num_states])
# for v_idx in range(num_vars):
#
#     # add messages from all factors sent to this variable
#     max_marginals[v_idx] = np.zeros(num_states)
#     for f_idx in ne_var[v_idx]:
#         max_marginals[v_idx] += msg_fv[(f_idx, v_idx)]
#     # print max_marginals[v_idx]
#
# # get MAP solution
# map_est = np.argmax(max_marginals, axis=1)
#
# # plot MAP estimate
# plt.imshow(map_est.reshape(h, w), interpolation='nearest');
# plt.gray()
# plt.show()

# 定义 alpha 值的范围
alpha_values = np.linspace(0, 1, 10)  # 在 0 和 1 之间生成 10 个值

# 创建一个新的图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # 创建一个 2x5 的子图网格

# 遍历所有的 alpha 值
for i, alpha_val in enumerate(alpha_values):
    # initialize factors (list of dictionaries), each factor comprises:
    #   vars: array of variables involved
    #   vals: vector/matrix of factor values
    factors = []

    # 添加一元因子
    for u in range(w):
        for v in range(h):
            # 从图像中获取观测值
            obs = img[v, u]

            # 创建一元因子，其中 'vars' 是变量的索引，'vals' 是对应的因子值
            unary_factor = {'vars': [v * w + u], 'vals': np.array([1 - obs, obs])}
            factors.append(unary_factor)

    # 添加二元因子
    E = alpha_val * np.array([[1, 0], [0, 1]])  # 二元因子的能量矩阵
    for u in range(w):
        for v in range(h):

            # 计算当前像素的索引
            var_index = v * w + u

            # 创建二元因子，其中 'vars' 是变量的索引，'vals' 是对应的因子值
            pairwise_factor = {'vars': [var_index, var_index + 1], 'vals': np.exp(E)}

            # 如果不是最后一列，添加到因子列表中
            if u < w - 1:
                factors.append(pairwise_factor)

            # 如果不是最后一行，添加到因子列表中
            if v < h - 1:
                pairwise_factor = {'vars': [var_index, var_index + w], 'vals': np.exp(E)}
                factors.append(pairwise_factor)

    # initialize all messages
    msg_fv = {}  # f->v messages (dictionary)
    msg_vf = {}  # v->f messages (dictionary)
    ne_var = [[] for i in range(num_vars)]  # neighboring factors of variables (list of list)

    # set messages to zero; determine factors neighboring each variable
    for [f_idx, f] in enumerate(factors):
        for v_idx in f['vars']:
            msg_fv[(f_idx, v_idx)] = np.zeros(num_states)  # factor->variable message
            msg_vf[(v_idx, f_idx)] = np.zeros(num_states)  # variable->factor message
            ne_var[v_idx].append(f_idx)  # factors neighboring variable v_idx

    # status message
    print("Messages initialized!")
    # run inference
    for it in range(30):

        # for all factor-to-variable messages do
        for [key, msg] in msg_fv.items():

            # shortcuts to variables
            f_idx = key[0]  # factor (source)
            v_idx = key[1]  # variable (target)
            f_vars = factors[f_idx]['vars']  # variables connected to factor
            f_vals = factors[f_idx]['vals']  # vector/matrix of factor values

            # unary factor-to-variable message
            if np.size(f_vars) == 1:
                msg_fv[(f_idx, v_idx)] = f_vals

            # pairwise factor-to-variable-message
            else:

                # if target variable is first variable of factor
                if v_idx == f_vars[0]:
                    msg_in = np.tile(msg_vf[(f_vars[1], f_idx)], (num_states, 1))
                    msg_fv[(f_idx, v_idx)] = (f_vals + msg_in).max(1)  # max over columns

                # if target variable is second variable of factor
                else:
                    msg_in = np.tile(msg_vf[(f_vars[0], f_idx)], (num_states, 1))
                    msg_fv[(f_idx, v_idx)] = (f_vals + msg_in.transpose()).max(0)  # max over rows

            # normalize
            msg_fv[(f_idx, v_idx)] = msg_fv[(f_idx, v_idx)] - np.mean(msg_fv[(f_idx, v_idx)])

        # for all variable-to-factor messages do
        for [key, msg] in msg_vf.items():

            # shortcuts to variables
            v_idx = key[0]  # variable (source)
            f_idx = key[1]  # factor (target)

            # add messages from all factors send to this variable (except target factor)
            # and send the result to the target factor
            msg_vf[(v_idx, f_idx)] = np.zeros(num_states)
            for f_idx2 in ne_var[v_idx]:
                if f_idx2 != f_idx:
                    msg_vf[(v_idx, f_idx)] += msg_fv[(f_idx2, v_idx)]

            # normalize
            msg_vf[(v_idx, f_idx)] = msg_vf[(v_idx, f_idx)] - np.mean(msg_vf[(v_idx, f_idx)])

    # calculate max-marginals (num_vars x num_states matrix)
    max_marginals = np.zeros([num_vars, num_states])
    for v_idx in range(num_vars):

        # add messages from all factors sent to this variable
        max_marginals[v_idx] = np.zeros(num_states)
        for f_idx in ne_var[v_idx]:
            max_marginals[v_idx] += msg_fv[(f_idx, v_idx)]
        # print max_marginals[v_idx]

    # get MAP solution
    map_est = np.argmax(max_marginals, axis=1)

    # 在子图中绘制 MAP 估计
    ax = axs[i // 5, i % 5]  # 选择子图
    ax.imshow(map_est.reshape(h, w), interpolation='nearest')  # 绘制图像
    ax.set_title(f'alpha = {alpha_val:.2f}')  # 设置标题

# 显示图形
plt.tight_layout()
plt.show()
