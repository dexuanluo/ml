import numpy as np
import random
from math import pi, exp

class pca(object):
    def __init__(self, data: np.array, k: int, centering = False):
        self.k = k
        if centering:
            self.data = (data - np.mean(data, axis = 0)).T
        else:
            self.data = data.T
        
        self.cov = (self.data @ self.data.T) / len(self.data[0])
    
    def train(self):
        val, vec = np.linalg.eig(self.cov)
        val = val.tolist()
        
        for i in range(len(val)):
            val[i] = (val[i], i)
        val.sort(reverse = True)
        newCov = np.empty((self.k, len(vec[0])))
        self.explainedVariance = val
        vec = vec.T
        for i in range(self.k):
            newCov[i] = vec[val[i][1]]
        self.directions = newCov
        self.newData = newCov @ self.data