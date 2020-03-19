import numpy as np
import random
from math import pi, exp

class fastmap(object):
    
    def __init__(self, disfunc, k):
        self.disfunc = disfunc
        self.k = k
        self.dim = len(disfunc)
        self.coordinates = np.zeros([k, self.dim])
    #O(k)
    def _getDis(self, a, b, index = 0):
        dis = self.disfunc[a][b]
        for i in range(index):
            dis = (dis**2 - \
            (self.coordinates[i][a] - \
             self.coordinates[i][b])**2)**0.5
        return dis
    ##O(kn)
    def _choose(self, index = 0):
        flag = True
        start = random.randint(0, self.dim - 1)
        last = start
        maxDis = 0
        maxIndex = None
        while flag:
            for i in range(self.dim):
                dis = self._getDis(start, i, index)
                if dis > maxDis:
                    maxDis = dis
                    maxIndex = (start, i)
            
            if maxIndex[1] == last:
                flag = False
            else:
                last = start
                start = maxIndex[1]
                maxDis = 0
                maxIndex = None
        return maxIndex
    #O(kn)
    def mapping(self, index = 0):
        if index >= self.k: return
        disfunc = self._getDis
        a, b = self._choose(index)
        for i in range(self.dim):
            self.coordinates[index][i] = (disfunc(a, i, index)**2 + \
            disfunc(a, b, index)**2 - disfunc(b, i, index)**2) / \
            (2 * disfunc(a, b, index))
        index += 1
        self.mapping(index)