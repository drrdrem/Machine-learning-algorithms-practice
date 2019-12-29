import numpy as np

def conv_layer(image, kernel, bias, stride=1):
    n_ker, n_col_ker, ker, _ = kernel.shape
    n_col, in_dim, _ = image.shape

    out_dim = int((in_dim-ker)/stride) + 1

    out = np.zeros((n_ker, out_dim))

    for cur_ker in range(n_ker):
        cur_j = j = 0

        while cur_j + ker <= in_dim:
            cur_i = i = 0
            while cur_i + ker <= in_dim:
                out[cur_ker, j, i] = np.sum(kernel[cur_ker]*image[:, cur_j:cur_j+ker, cur_i:cur_i+ker]) + bias[cur_ker]
                cur_i += stride
                i += 1
            cur_j += stride
            j += 1

    return out

def max_pool_layer(image, f=2, s=2):
    n_col, h_, w_ = image.shape

    h = int((h_ - f)/s) + 1
    w = int((w_ - f)/s) + 1

    out = np.zeros((n_col, h, w))

    for cur_col in range(n_col):
        cur_j = j = 0

        while cur_j + f <= h_:
            cur_i = i = 0

            while cur_i + f <= w_:
                out[cur_col, j, i] = np.max(image[cur_col, cur_j:cur_j+f, cur_i+f])
                cur_i += s
                i += 1
            cur_j += s
            j += 1

    return out