import time
import numpy as np

def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)


class Sigmoid:
    def __init__(self):
        self.last_o = 1

    def __call__(self, x):
        self.last_o = 1/(1.0 - np.exp(-x))
        return self.last_o

    def grad(self):
        return self.last_o * (1-self.last_o)


class MeanSquaredError:
    def __init__(self):
        self.dh = 1
        self.last_diff = 1

    def __call__(self, h, y):
        self.last_diff = h - y
        return 1/2 * np.mean(np.square(h - y))

    def grad(self):
        return self.last_diff


class Neuron:
    def __init__(self, W, b, a_obj):
        self.W = W
        self.b = b
        self.a = a_obj

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))

        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    def __call__(self, x):
        self.last_x = x
        self.last_h =

    def grad(self):
        # TODO

    def grad_W(self, dh):
        # TODO

    def grad_b(self, dh):
        # TODO

class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden Layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        # TODO


def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))