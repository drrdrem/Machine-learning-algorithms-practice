import numpy as np

def conv_layer_backprop(dconv, conv_in, kernel, stride):

    n_ker, n_col, ker = kernel.shape
    _, orig_dim, _ = conv_in.shape

    dout = np.zeros(conv_in.shape)
    dkernel = np.zeros(kernel.shape)
    dbias = np.zeros((n_ker, 1))

    for cur_ker in range(n_ker):
        cur_j = j = 0

        while cur_j + ker <= orig_dim:
            cur_i = i = 0

            while cur_i + ker <= orig_dim:
                dkernel[cur_ker] += dconv[cur_ker, j, i]*ker[cur_ker]
                cur_i += stride
                i += 1
            cur_j += stride
            j += 1
        dbias[cur_ker] = np.sum(dconv[cur_ker])

    return dout, dkernel, dbias

def n_nan_argmax(array):
    return np.unravel_index(np.nanargmax(array), array.shape)

def max_pool_layer_backprop(dpool, origin, f, s):
    n_col, orig_dim, _ = origin.shape

    dout = np.zeros(origin.shape)

    for cur_col in range(n_col):
        cur_j = j = 0

        while cur_j + f <= orig_dim:
            cur_i = i = 0

            while cur_i + f <= orig_dim:
                a, b = n_nan_argmax(origin[cur_col, cur_j:cur_j+f, cur_i:cur_i+f])
                dout[cur_col, cur_j+a, cur_i+b] = dpool[cur_col, j, i]

                cur_i += s
                i += 1
            
            cur_j += s
            j += 1

    return dout