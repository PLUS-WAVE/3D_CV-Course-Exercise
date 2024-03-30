# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


# plot function
# input: Nx3 matrix of values & title string
def plot(vals, title=''):
    plt.close()
    vals /= np.tile(np.sum(vals, 1), (3, 1)).transpose()
    f, axarr = plt.subplots(1, 10, figsize=(10, 2))
    plt.suptitle(title, fontsize=16, fontweight='bold')
    for i in range(vals.shape[0]):
        axarr[i].barh([0, 1, 2], np.array([1, 1, 1]), color='white', edgecolor='black', linewidth=2)
        axarr[i].barh([0, 1, 2], vals[i], color='red')
        axarr[i].axis('off')
    plt.show()


if __name__ == '__main__':
    # unary: Nx3 matrix specifying unary likelihood of each state
    unary = np.array([[0.7, 0.1, 0.2], [0.7, 0.2, 0.1], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1],
                      [0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1],
                      [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    # pairwise: 3x3 matrix specifying transition probabilities (rows=t -> columns=t+1)
    pairwise = np.array([[0.8, 0.2, 0.0], [0.2, 0.6, 0.2], [0.0, 0.2, 0.8]])

    # plot unaries
    # plot(unary, 'Unary')

    # model parameters (number of variables/states)
    [num_vars, num_states] = unary.shape

    # compute messages
    msg = np.zeros([num_vars - 1, num_states])  # (num_vars-1) x num_states matrix
    for i in range(num_vars - 2, -1, -1):
        if i == num_vars - 2:
            # 计算倒数第二个到最后一个的消息
            msg[i] = np.max(pairwise * unary[i + 1, :], 1)
        else:
            # 计算其他的消息
            msg[i] = np.max(pairwise * unary[i + 1, :] * msg[i + 1], 1)

    # calculate max-marginals (num_vars x num_states matrix) and MAP estimates (num_vars x 1 matrix)
    max_marginals = np.zeros([num_vars, num_states])
    map = np.zeros(num_vars, dtype=int)
    for i in range(num_vars):
        if i == 0:
            max_marginals[i, :] = msg[i, :]
        if i == num_vars - 1:
            max_marginals[i, :] = pairwise[map[i - 1], :] * unary[i, :]
        else:
            max_marginals[i, :] = pairwise[map[i - 1], :] * unary[i, :] * msg[i, :]
        map[i] = np.argmax(max_marginals[i, :])

    # plot max-marginals
    plot(max_marginals, 'Max Marginals')

    # print MAP state
    print("MAP Estimate:")
    print(map)
