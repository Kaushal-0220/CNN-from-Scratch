from . import library as lib
from .activation import activation_softmax

class loss_categorical_cross_entropy:
    def calculate(self, outputs, y):
        samples = len(outputs)
        y_pred_clipped = lib.cp.clip(outputs, 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        else:
            correct_confidences = lib.cp.sum(y_pred_clipped * y, axis=1)
        return lib.cp.mean(-lib.cp.log(correct_confidences))

class activation_softmax_loss_categorical_cross_entropy:
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = loss_categorical_cross_entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = lib.cp.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples