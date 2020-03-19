import numpy as np
import pandas as pd
import random
from math import pi, exp


class gmm(object):
    
    ##Define the data I will be sotring in my class and randomly 
    ##initiate centroids and covariance matrix
    def __init__(self, data: pd.DataFrame, nclusters: int, mu = None):
        
        self.data = np.matrix(data.to_numpy())
        self.c = nclusters
        if not mu:
            self.mu = \
            np.array([self.data[random.randint(0,len(self.data) - 1)]\
                      for i in range(nclusters)])
        else:
            self.mu = mu
        self.pi = np.empty([self.c])
        for i in range(len(self.pi)):
            self.pi[i] = random.randint(1,50)
        self.pi /= sum(self.pi)
        
        self.membership = np.empty([len(self.data), self.c])
        self.cov = [np.cov(self.data.T) * self.pi[i] \
                    for i in range(nclusters)]
        
    
    #Calculate the partial membership for each points.
    def _eStep(self):

        for c in range(self.c):
            mu = self.mu[c]
            pie = self.pi[c]
            cov = self.cov[c]
            det = np.linalg.det(cov)
            inv = np.linalg.inv(cov)
            
            for i in range(len(self.data)):
                normalized = self.data[i] - mu
                prob = exp(-1/2 * normalized @ inv @ normalized.T) /\
                (((2 * pi)**self.data.ndim) * det)**0.5
                self.membership[i][c] = prob * pie
                
        for i in range(len(self.data)):
            total = sum(self.membership[i])
            for j in range(self.c):
                self.membership[i][j] /= total

                
    ##Re-estimate mean, amplitude and covariance matrix   
    def _mStep(self):
        
        totalMembership = np.sum(self.membership, axis = 0)
        for c in range(self.c):
            self.mu[c] = self.data.T @ gmm_mod.membership.T[c] / \
            totalMembership[c]
            self.pi[c] = totalMembership[c] / len(self.data)
            for i in range(len(self.data)):
                
                normalized = (self.data[i] - self.mu[c]) * \
                self.membership[i][c]
                normalized = normalized.T @ normalized
                if i == 0:
                    self.cov[c] = normalized
                else:
                    self.cov[c] += normalized
            
            self.cov[c] /= totalMembership[c]

    
    def train(self, maxIter = 10, threshold = 0):
        try:
            
            for i in range(maxIter):
                last = np.sum(self.membership, axis = 0)
                self._eStep()
                self._mStep()
                post = np.sum(self.membership, axis = 0)
                dif = sum(abs(last - post))
                if dif <= threshold:
                    print("It took " + str(i) + " iterations to converge")
                    break
                elif i == maxIter - 1:
                    print("Failed to converge withhin the max iteration")
        except:
            print("Failed to converge due to error")