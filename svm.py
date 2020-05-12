import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class polykernel(object):
    def __init__(self):
        ##structure for kernel
        self.val = None
        self.w = None
        self.b = None
    
    def fit(self, x, d = 2, c = 0):
        ##main function
        self.val = (x @ x.T + c)**d
        return self.val

    def getWeight(self, x, y, alphas):
        ##weight in kernel space
        self.w = (y * alphas).T @ np.square(x)
        return self.w
    
    def getOffset(self, x, y, sv):
        self.b = np.sum(np.sum(self.w * np.square(x[sv]), axis = 1)\
                        - y[sv].T) / len(y[sv])
        return self.b

class linearkernel(object):
    def __init__(self):
        self.val = None
        self.w = None
        self.b = None
    
    def fit(self, x):
        self.val = x @ x.T
        return self.val
    
    def getWeight(self, x, y, alphas):
        self.w = (y * alphas).T @ x
        return self.w
    
    def getOffset(self, x, y, sv):
        self.b = np.sum(np.sum(self.w * x[sv], axis = 1) - y[sv].T) / \
        len(y[sv])
        return self.b


class svm(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def train(self, kernel = None, alphasthreshold = 1e-4, **kwargs):
        self.kernel = kernel
        x2 = kernel.fit(x = self.x, **kwargs)
        ##Preparation
        i, j = self.x.shape
        Q = (self.y @ self.y.T) * x2
        P = cvxopt_matrix(Q)
        q = cvxopt_matrix(-np.ones((i, 1)))
        G = cvxopt_matrix(-np.eye(i))
        h = cvxopt_matrix(np.zeros(i))
        A = cvxopt_matrix(self.y.T)
        b = cvxopt_matrix(np.zeros(1))
        ##Quadratic programming solver
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10
        ##producing result and support vectors
        self.res = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(self.res['x'])
        self.sv = (self.alphas >= alphasthreshold).flatten()
        ###Weight and Offset in kernel space
        self.w = kernel.getWeight(self.x, self.y, self.alphas)
        self.b = kernel.getOffset(self.x, self.y, self.sv)


