from . import library as lib

# ReLU Activation Function
class activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = lib.cp.maximum(0, inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


# Softmax Activation Function
class activation_softmax:
    def forward(self, inputs):
        exp_values = lib.cp.exp(inputs - lib.cp.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / lib.cp.sum(exp_values, axis=1, keepdims=True)
        return self.outputs