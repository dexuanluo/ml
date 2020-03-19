class logistic(object):
    def __init__(self, x, y):
        self.x = np.append(np.array([1 for i in range(len(x[0]))]), \
                           x).reshape(len(x) + 1, len(x[0])).T
        self.y = y
                
        self.weight = np.array([random.random() * random.choice([-1, 1])\
                                for i in range(len(x) + 1)])
    
    def _h(self, xi):
        return 1 / (1 + e ** -(self.weight.T @ xi))
    
    def _derCost(self, xi, yi):
        return (self._h(xi) - yi) * xi
    
    def _cost(self, xi, yi):
        return -yi * log(self._h(xi)) - (1 - yi) * log(1 - self._h(xi))
    
    def train(self, maxIter, lr, tolerance, batch = None):
        self.entropy = []
        for iterations in range(maxIter):
            gradient = np.zeros(len(self.x[0]))
            entropy = 0
            if not batch:
                for xi, yi in zip(self.x, self.y):
                    entropy += self._cost(xi, yi)
                    gradient += self._derCost(xi, yi)
                entropy /= len(self.x)
                self.weight -= lr * (1 / len(self.x))* gradient
                self.entropy.append(entropy)
            else:
                sample = random.sample(range(len(self.x)), batch)
                for i in range(batch):
                    xi = self.x[sample[i]]
                    yi = self.y[sample[i]]
                    entropy += self._cost(xi, yi)
                    gradient += self._derCost(xi, yi)
                entropy /= batch
                self.weight -= lr * (1 / batch)* gradient
                self.entropy.append(entropy)
            
            if entropy <= tolerance:
                print('It took ' + str(iterations + 1) + ' iterations to converge')
                return
        print('The algorithm did not converge')
        return
    
    def fit(self, threshold = 0.5):
        self.prod = self._h(self.x.T)
        self.pred = []
        self.threshold = threshold
        acc = 0
        for prod, y in zip(self.prod, self.y):
            if prod < threshold:
                self.pred.append(0)
            else:
                self.pred.append(1)
            if self.pred[-1] == y:
                acc += 1
        self.acc = acc / len(self.x)
        return