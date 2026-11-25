from . import library as lib

#VECTORIZATION
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = lib.cp.repeat(lib.cp.arange(field_height), field_width)
    i0 = lib.cp.tile(i0, C)
    i1 = stride * lib.cp.repeat(lib.cp.arange(out_height), out_width)
    j0 = lib.cp.tile(lib.cp.arange(field_width), field_height * C)
    j1 = stride * lib.cp.tile(lib.cp.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = lib.cp.repeat(lib.cp.arange(C), field_height * field_width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = lib.cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = lib.cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    lib.cp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# CONVOLUTIONAL LAYER
class layer_convolution:
    def __init__(self, in_channels, num_filters, filter_size=3, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        self.weights = lib.cp.random.randn(num_filters, in_channels, filter_size, filter_size) * lib.cp.sqrt(2.0 / (in_channels * filter_size * filter_size))
        self.biases = lib.cp.zeros((num_filters, 1))

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        F, _, HH, WW = self.weights.shape
        H_out = int(1 + (H + 2 * self.padding - HH) / self.stride)
        W_out = int(1 + (W + 2 * self.padding - WW) / self.stride)

        self.inputs_col = im2col_indices(inputs, HH, WW, padding=self.padding, stride=self.stride)
        weights_row = self.weights.reshape(F, -1)

        out = lib.cp.dot(weights_row, self.inputs_col) + self.biases
        out = out.reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)
        self.outputs = out
        return self.outputs

    def backward(self, dvalues):
        F, C, HH, WW = self.weights.shape
        dvalues_reshaped = dvalues.transpose(1, 2, 3, 0).reshape(F, -1)

        self.dweights = lib.cp.dot(dvalues_reshaped, self.inputs_col.T).reshape(self.weights.shape)
        self.dbiases = lib.cp.sum(dvalues_reshaped, axis=1, keepdims=True)

        weights_row = self.weights.reshape(F, -1)
        dinputs_col = lib.cp.dot(weights_row.T, dvalues_reshaped)
        self.dinputs = col2im_indices(dinputs_col, self.inputs.shape, HH, WW, padding=self.padding, stride=self.stride)
        return self.dinputs


# POOLING LAYER
class layer_pooling:
    def __init__(self, pool_size=(2, 2), stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        ph, pw = self.pool_size
        x_reshaped = inputs.reshape(N, C, H // ph, ph, W // pw, pw)
        self.outputs = x_reshaped.max(axis=(3, 5))
        return self.outputs

    def backward(self, dvalues):
        N, C, H, W = self.inputs.shape
        ph, pw = self.pool_size
        x_reshaped = self.inputs.reshape(N, C, H // ph, ph, W // pw, pw)
        out = x_reshaped.max(axis=(3, 5), keepdims=True)
        mask = (x_reshaped == out)
        dvalues_reshaped = dvalues[:, :, :, None, :, None]
        dinputs = mask * dvalues_reshaped
        self.dinputs = dinputs.reshape(self.inputs.shape)
        return self.dinputs


# FLATTEN LAYER
class layer_flatten:
    def forward(self, inputs):
        self.orig_shape = inputs.shape
        self.outputs = inputs.reshape(inputs.shape[0], -1)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.orig_shape)
        return self.dinputs


# DENSE LAYER
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * lib.cp.random.randn(n_inputs, n_neurons)
        self.biases = lib.cp.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = lib.cp.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues):
        self.dweights = lib.cp.dot(self.inputs.T, dvalues)
        self.dbiases = lib.cp.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = lib.cp.dot(dvalues, self.weights.T)
        return self.dinputs


# DROPOUT LAYER
class layer_dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = lib.cp.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        return self.dinputs