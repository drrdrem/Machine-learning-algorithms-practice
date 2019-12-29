import numpy as np
def perceptron(data):
    x = np.array(data[['x1', 'x2']])
    y = np.array(data[['y']])
    a = [1, 1]
    check = True
    while check:
        for i in range(data.shape[0]):
            if np.inner(y[i]*a, x[i])<0: a += y[i]*x[i]
            check = False
            for i in range(data.shape[0]):
                if np.inner(y[i]*a, x[i])<0: check = True
    return check