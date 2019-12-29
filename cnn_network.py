from cnn_forward import *
from cnn_backprop import *

def xoss_entropy(probs, labels):
    return -np.sum(labels*np.log(probs))

def soft_max(preds):
    softmax = np.exp(preds)
    return softmax/np.sum(softmax)

def cnn_network(image, label, params, conv_stride, pool_f, pool_s):
    [ker1, ker2, w3, w4, bia1, bia2, bia3, bia4] = params

    conv1 = conv_layer(image, ker1, bia1, conv_stride)
    conv1[conv1<=0] = 0

    conv2 = conv_layer(conv1, ker2, bia2, conv_stride)
    conv2[conv2<=0] = 0

    pool1 = max_pool_layer(conv2, pool_f, pool_s)

    (nker2, dim2, _) = pool1.shape
    fc1 = pool1.reshape((nker2*dim2*dim2, 1))

    z = w3.dot(fc1) + bia3
    z[z<=0] = 0

    output = w4.dot(z) + bia4
    probs = soft_max(output)

    loss = xoss_entropy(probs, label)
    doutput = probs - label
    dw4 = doutput.dot(z.T)
    dbia4 = np.sum(doutput, axis=1).reshape(bia4.shape)

    dz = w4.T.dot(doutput)
    dz[z<=0] = 0
    dw3 = dz.dot(fc1.T)
    dbia3 = np.sum(dz, axis=1).reshape(bia3.shape)

    dfc1 = w3.T.dot(dz)
    dpool1 = dfc1.reshape(pool1.shape)

    dconv2 = max_pool_layer_backprop(dpool1, conv2, pool_f, pool_s)
    dconv2[conv2<=0] = 0

    dconv1, dker2, dbia2 = conv_layer_backprop(dconv2, conv1, ker2, conv_stride)
    dimage, dker1, dbia1 = conv_layer_backprop(dconv1, image, ker1, conv_stride)

    gradients = [dker1, dker2, dw3, dw4, dbia1, dbia2, dbia3, dbia4]

    return gradients, loss