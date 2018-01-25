from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        c, height, width = input_dim
        self.params['W1'] = np.random.randn(num_filters, c, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.randn(height * width * num_filters // 4, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        ar_out, ar_cache = affine_relu_forward(crp_out, W2, b2)
        scores, a_cache = affine_forward(ar_out, W3, b3)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        for i in range(3):
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W%d' % (i + 1)]))

        dx3, grads['W3'], grads['b3'] = affine_backward(dscores, a_cache)
        dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, ar_cache)
        dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, crp_cache)

        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class MyCNN(object):
    """
    [[conv - bn - relu] x 2] - pool] x M - [affine - bn - relu - dropout] x N - affine - softmax
    """
    def __init__(self, filters, hidden_dims, input_dim=(3, 32, 32), num_classes=10,
                 weight_scale=1e-3, dropout=0, reg=0.0, dtype=np.float32):

        self.reg = reg
        self.dtype = dtype
        self.use_dropout = dropout > 0
        self.params = {}

        self.num_conv = len(filters['num'])
        self.num_fc = len(hidden_dims) + 1
        self.num_layers = self.num_conv + self.num_fc
        num_pool = self.num_conv / 2
        channel, height, width = input_dim
        fc_h = height / 2 ** num_pool
        fc_w = width / 2 ** num_pool
        w_channel = channel
        for i, mf in enumerate(filters['num']):
            self.params['W%d' % (i + 1)] = np.random.randn(
                mf, w_channel, filters['sizes'][i], filters['sizes'][i]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(mf)
            self.params['gamma%d' % (i + 1)] = np.ones(mf)
            self.params['beta%d' % (i + 1)] = np.zeros(mf)
            w_channel = mf

        layer_input_dim = int(w_channel * fc_h * fc_w)
        for i, hd in enumerate(hidden_dims):
            self.params['W%d' % (i + self.num_conv + 1)] = np.random.randn(layer_input_dim, hd) * weight_scale
            self.params['b%d' % (i + self.num_conv + 1)] = np.zeros(hd)
            self.params['gamma%d' % (i + self.num_conv + 1)] = np.ones(hd)
            self.params['beta%d' % (i + self.num_conv + 1)] = np.zeros(hd)
            layer_input_dim = hd
        self.params['W%d' % self.num_layers] = np.random.randn(layer_input_dim, num_classes) * weight_scale
        self.params['b%d' % self.num_layers] = np.zeros(num_classes)

        self.bn_params = [{'mode': 'train'} for i in range(self.num_conv + self.num_fc - 1)]

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        if self.use_dropout:
            self.dropout_param['mode'] = mode

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        lay_input = X
        lay_cache = {}
        pool_cache = {}
        dp_cache = {}
        for lay in range(1, self.num_layers + 1):
            if lay <= self.num_conv:
                conv_param = {'stride': 1, 'pad': (self.params['W%d' % lay].shape[2] - 1) // 2}
                lay_input, lay_cache[lay] = conv_bn_relu_forward(
                    lay_input,
                    self.params['W%d' % lay],
                    self.params['b%d' % lay],
                    self.params['gamma%d' % lay],
                    self.params['beta%d' % lay],
                    conv_param,
                    self.bn_params[lay - 1],
                )
                if lay % 2 == 0:
                    lay_input, pool_cache[lay] = max_pool_forward_fast(lay_input, pool_param)

            elif lay <= self.num_layers - 1:
                    lay_input, lay_cache[lay] = affine_bn_relu_forward(
                        lay_input,
                        self.params['W%d' % lay],
                        self.params['b%d' % lay],
                        self.params['gamma%d' % lay],
                        self.params['beta%d' % lay],
                        self.bn_params[lay - 1]
                    )
                    if self.use_dropout:
                        lay_input, dp_cache[lay] = dropout_forward(lay_input, self.dropout_param)
            else:
                    scores, lay_cache[lay] = affine_forward(
                        lay_input,
                        self.params['W%d' % lay],
                        self.params['b%d' % lay]
                    )

        if mode == 'test':
            return scores

        grads = {}
        loss, dscores = softmax_loss(scores, y)
        dout = dscores
        dgamma, dbeta = 0, 0
        for lay in range(self.num_layers, 0, -1):
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W%d' % lay]))
            if lay <= self.num_conv:
                if lay % 2 == 0:
                    dout = max_pool_backward_fast(dout, pool_cache[lay])
                dout, dw, db, dgamma, dbeta = conv_bn_relu_backward(dout, lay_cache[lay])
            elif lay <= self.num_layers - 1:
                if self.use_dropout:
                    dout = dropout_backward(dout, dp_cache[lay])
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, lay_cache[lay])
            else:
                dout, dw, db = affine_backward(dout, lay_cache[lay])

            grads['W%d' % lay] = dw + self.reg * self.params['W%d' % lay]
            grads['b%d' % lay] = db
            if lay < self.num_layers:
                grads['gamma%d' % lay] = dgamma
                grads['beta%d' % lay] = dbeta

        return loss, grads
