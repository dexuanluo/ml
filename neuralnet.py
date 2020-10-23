import numpy as np
import types


##A doubly linked list as a basic building block of the network.
class layer(object):
    def __init__(self, shape = None, \
                 inputLayer = False, data = None):
        if data is not np.array:
            data = np.array(data)
        
        if inputLayer:
            self.input = True
            self.output = data
            self.shape = len(data[0])
            
        else:
            self.input = False
            self.output = None
            self.shape = None
        self.weight = None
        self.last = None
        self.next = None


class neuralnet(object):
    
    def __init__(self, data):
        self.head = layer(inputLayer = True, data = data)
        self.tail = self.head
        
        
###Sigmoid activation function
    def sigmoid(self, x, derivative = False):
        if derivative:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1 / (1 + np.exp(-x))
###Add layer  
    def addLayer(self, shape):
        self.tail.next = layer(shape)
        self.tail.next.last = self.tail
        self.tail = self.tail.next
        self.tail.weight = 2 * \
        np.random.random((self.tail.last.shape, shape)) - 1
        self.tail.shape = shape
        return self.tail
###Remove a layer
    def removeLayer(self, node):
        last = node.last
        last.next = node.next.next
        node.last = None
        node.next = None
        return last
###MSE loss function
    def _sqloss(self, x, y, derivative = False):
        if derivative:
            return x * (x - y)
        return np.square(x - y)
### Forwardpropagation
    def _forward(self, activation = "sigmoid"):
        
        cur = self.head
        if isinstance(activation, types.FunctionType):
            acfunc = activation
        if activation == "sigmoid":
            acfunc = self.sigmoid
        while cur.next:
            cur.next.output = acfunc((cur.output @ cur.next.weight))
            cur = cur.next

###Backwardpropagation
    def _backward(self, lable, lr = 1, loss = "square",\
                  activation = "sigmoid"):
        
        if isinstance(activation, types.FunctionType):
            acfunc = activation
        if activation == "sigmoid":
            acfunc = self.sigmoid
        if loss == "square":
            lossfunc = self._sqloss
        if isinstance(loss, types.FunctionType):
            lossfunc = loss
        cur = self.tail
        delta = lossfunc(self.tail.output, lable, derivative = True)
        gradient = 2 * lr * np.mean((cur.last.output * delta), axis = 0)
        gradient.shape = (cur.last.shape, cur.shape)
        while cur.last:
            cur.weight -= gradient
            delta = (1 - np.square(cur.last.output)) * \
            (delta @ cur.weight.T)
            gradient = lr * np.mean((cur.last.output * delta), axis = 0)
            cur = cur.last
        
###Main training loop   
    def train(self, lable, maxIter = 1000, lr = 0.1, **kwargs):
        for _ in tqdm(range(maxIter)):
            self._forward()
            self._backward(lable, lr = lr)
        print("Error: " + str(np.mean(self._sqloss(self.tail.output, \
                                                   lable))))

###Predict testing data
    def predict(self, test, activation = "sigmoid"):
        cur = self.head
        res = test
        if isinstance(activation, types.FunctionType):
            acfunc = activation
        if activation == "sigmoid":
            acfunc = self.sigmoid
        while cur.next:
            res = acfunc((res @ cur.next.weight))
            cur = cur.next
    
        return res