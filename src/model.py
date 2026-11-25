from .library import *
from .activation import *
from .data import *
from .layers import *
from .loss import *
from .optimizer import *


class CNN:

    def __init__(self, num_classes=4):
        self.num_classes = num_classes

        self.conv1 = layer_convolution(in_channels=3, num_filters=32, filter_size=3, stride=1, padding=1)
        self.relu1 = activation_relu()
        self.pool1 = layer_pooling(pool_size=(2, 2), stride=2)

        self.conv2 = layer_convolution(in_channels=32, num_filters=64, filter_size=3, stride=1, padding=1)
        self.relu2 = activation_relu()
        self.pool2 = layer_pooling(pool_size=(2, 2), stride=2)

        self.conv3 = layer_convolution(in_channels=64, num_filters=128, filter_size=3, stride=1, padding=1)
        self.relu3 = activation_relu()
        self.pool3 = layer_pooling(pool_size=(2, 2), stride=2)

        self.flatten = layer_flatten()

        self.dense1 = layer_dense(128 * 16 * 16, 256)
        self.relu4 = activation_relu()
        self.dropout1 = layer_dropout(0.5)

        self.dense2 = layer_dense(256, num_classes)
        self.loss_activation = (activation_softmax_loss_categorical_cross_entropy())
        self.optimizer = optimizer_adam(learning_rate=0.001, decay=1e-5)


    def forward(self, X, training=True):
        self.conv1.forward(X)
        self.relu1.forward(self.conv1.outputs)
        self.pool1.forward(self.relu1.outputs)

        self.conv2.forward(self.pool1.outputs)
        self.relu2.forward(self.conv2.outputs)
        self.pool2.forward(self.relu2.outputs)

        self.conv3.forward(self.pool2.outputs)
        self.relu3.forward(self.conv3.outputs)
        self.pool3.forward(self.relu3.outputs)

        self.flatten.forward(self.pool3.outputs)

        self.dense1.forward(self.flatten.outputs)
        self.relu4.forward(self.dense1.outputs)

        if training:
            self.dropout1.forward(self.relu4.outputs)
            dense_input = self.dropout1.outputs
        else:
            dense_input = self.relu4.outputs

        self.dense2.forward(dense_input)

        return self.dense2.outputs


    def backward(self, y_batch):
        self.loss_activation.backward(self.loss_activation.outputs, y_batch)
        self.dense2.backward(self.loss_activation.dinputs)
        
        self.dropout1.backward(self.dense2.dinputs)
        self.relu4.backward(self.dropout1.dinputs)
        self.dense1.backward(self.relu4.dinputs)

        self.flatten.backward(self.dense1.dinputs)

        self.pool3.backward(self.flatten.dinputs)
        self.relu3.backward(self.pool3.dinputs)
        self.conv3.backward(self.relu3.dinputs)

        self.pool2.backward(self.conv3.dinputs)
        self.relu2.backward(self.pool2.dinputs)
        self.conv2.backward(self.relu2.dinputs)

        self.pool1.backward(self.conv2.dinputs)
        self.relu1.backward(self.pool1.dinputs)
        self.conv1.backward(self.relu1.dinputs)


    def update_params(self):
        self.optimizer.pre_update_params()

        self.optimizer.update_params(self.conv1)
        self.optimizer.update_params(self.conv2)
        self.optimizer.update_params(self.conv3)

        self.optimizer.update_params(self.dense1)
        self.optimizer.update_params(self.dense2)

        self.optimizer.post_update_params()


    def train_step(self, X_batch, Y_batch):
        self.forward(X_batch, training=True)
        loss = self.loss_activation.forward(self.dense2.outputs, Y_batch)
        predictions = lib.cp.argmax(self.loss_activation.outputs, axis=1)
        correct = lib.cp.sum(predictions == Y_batch)
        self.backward(Y_batch)
        self.update_params()

        return loss, correct

    def predict(self, X):
        logits = self.forward(X, training=False)
        probabilities = activation_softmax().forward(logits)
        predictions = lib.cp.argmax(probabilities, axis=1)

        return predictions