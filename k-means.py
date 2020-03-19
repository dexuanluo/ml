import numpy as np
import pandas as pd
import random
from math import pi, exp


class kmeans(object):
    
    def __init__(self, x: pd.DataFrame, k: int):
        self.data = x
        self.k = k
        self.centroids = None
        self.labels = None
#Initialize centroids by randomly choosing a data point as the cnetroids  
    def _initialization(self):
        self.centroids = data.sample(n = self.k)
        self._clustering()
#Clustering data based on assigned labels
    def _clustering(self):
        k = len(self.centroids)
        l = len(self.data)
        labels = [None]*l
        for i in range(l):
            minDist = float("Inf")
            minIndex = 0
            for j in range(k):
                dist = 0
                for a in range(self.data.ndim):
                    dist += (self.data.iloc[i].iloc[a] - \
                             self.centroids.iloc[j].iloc[a])**2
                dist = dist**0.5
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            labels[i] = minIndex
        
        self.labels = labels
#Recompute the centroids of model
    def _recomputeCentroids(self):
        
        count = [0] * self.k
        for i in range(len(self.data)):
            label = self.labels[i]
            count[label] += 1
            for k in range(self.data.ndim):
                self.centroids.iloc[label].iloc[k] \
                += self.data.iloc[i].iloc[k]
        for i in range(self.k):
            for k in range(self.data.ndim):
                self.centroids.iloc[i].iloc[k] /= count[i]
        
#Train the model by repeating the previous method
#maxIter is the maximum iteration
# tolerance is the allowed missclassified data points
    def train(self, maxIter = 50, tolerance = 0):
        self._initialization()
        l = len(self.labels)
        for i in range(maxIter):
            last = self.labels.copy()
            self._recomputeCentroids()
            self._clustering()
            diff = 0
            for j in range(l):
                if last[j] != self.labels[j]:
                    diff += 1
            if diff <= tolerance:
                print("Done, iterated " + str(i) + " times to converge.")
                break
            elif i == l - 1:
                print('The algorithm did not converge')