import numpy as np
import random
from math import pi, exp, e, log

class perceptron(object):
    def __init__(self, x, y):
        self.x = np.append(np.array([1 for i in range(len(x[0]))]), \
                           x).reshape(len(x) + 1, len(x[0]))
        self.y = y
        self.weight = np.array([random.random() * random.choice([-1, 1])\
                                for i in range(len(x) + 1)])
    
    def train(self, lr, maxIter, tolerance):
        self.bestAcc = 0
        self.error = []
        self.bestWeight = None
        for k in range(maxIter):
            res = self.weight.T @ self.x
            error = []
            for i in range(len(res)):
                if (res[i] >= 0 and self.y[i] == -1) \
                or (res[i] < 0 and self.y[i] == 1):
                    error.append([self.x[j][i] for j in range(len(self.x))])
            acc = 1 - (len(error) / len(self.x[0]))
            self.error.append(1 - acc)
            if 1 - acc > tolerance:
                learnFrom = np.array(random.choice(error))
                if self.weight.T @ learnFrom < 0:
                    self.weight += np.array(learnFrom)*lr
                else:
                    self.weight -= np.array(learnFrom)*lr
    
                if acc > self.bestAcc:
                    self.bestAcc = acc
                    self.bestWeight = self.weight
            
            else:
                if acc > self.bestAcc:
                    self.bestAcc = acc
                    self.bestWeight = self.weight
                print('It took ' + str(k + 1) + ' iterations to converge')
                self.iter = k + 1
                return
                
        self.iter = k + 1
        print('The algorithm did not converge')
        return

