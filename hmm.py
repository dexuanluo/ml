import numpy as np

if __name__ == "__main__":
    graph = \
    np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    obstacles = set([22, 23, 24, 25, 26, 32, 36, 42, 46, 52, 56, 62, 66])
###Initial probability for each state
    init_prob = np.array([1/ 87 for i in range(87)])
    counter = 0
    encoder = {}
    decoder = {}
    for i in range(10):
        for j in range(10):
            if (10 * i + j) not in obstacles:
                encoder[10 * i + j] = counter
                decoder[counter] = 10 * i + j
                counter += 1
##Transition matrix
    trans_prob = [[0] * 87 for _ in range(87)]
    
    for i in range(10):
        for j in range(10):
            if graph[i][j] == 1:
                count = 0
                res = []
                pi = i + 1
                ni = i - 1
                pj = j + 1
                nj = j - 1
                if pi < 10 and graph[pi][j] == 1:
                    res.append((pi, j))
                    count += 1
                if ni >= 0 and graph[ni][j] == 1:
                    res.append((ni, j))
                    count += 1
                if pj < 10 and graph[i][pj] == 1:
                    res.append((i, pj))
                    count += 1
                if nj >= 0 and graph[i][nj] == 1:
                    res.append((i, nj))
                    count += 1
                for a, b in res:
                    trans_prob[encoder[10 * i + j]][encoder[10 * a + b]] = 1 / count
            
    trans_prob =  np.array(trans_prob)
    
#the maximum possible value for d is 16.5
#The input emission probability takes input shape as 
#(# of types of observation, #number of classes of each type of observation, # of states)
    emission_prob = [np.zeros([166,87]) for _ in range(4)]
    emission_prob = np.array(emission_prob)
    for i in range(10):
        for j in range(10):
            if 10*i + j not in obstacles:
                d = [np.sqrt(i**2 + j**2),\
                     np.sqrt(i**2 + (9 - j)**2),\
                     np.sqrt((9 - i)**2 + j**2),\
                     np.sqrt((9 - i)**2 + (9 - j)**2)]
                errors = []
                for k in range(len(emission_prob)):
                    lower = 0.7 * d[k]
                    upper = 1.3 * d[k]
                    lower -= lower % 0.1
                    upper -= upper % 0.1
                    lower = int(lower * 10)
                    upper = int(upper * 10)
                    errors.append([i for i in range(lower, upper + 1)])
                for k in range(len(emission_prob)):
                    for error in errors[k]:
                        emission_prob[k][error][encoder[10*i + j]] = 1
    np.seterr(all = 'ignore')
    for k in range(4):
        colSum = np.sum(emission_prob[k], axis = 0)
        for i in range(87):
            emission_prob[k][i] /= colSum
    
    
    
    observations = [[6.3, 5.9, 5.5, 6.7],\
                    [5.6, 7.2, 4.4, 6.8],\
                    [7.6, 9.4, 4.3, 5.4],\
                    [9.5, 10.0, 3.7, 6.6],\
                    [6.0, 10.7, 2.8, 5.8],\
                    [9.3, 10.2, 2.6, 5.4],\
                    [8.0, 13.1, 1.9, 9.4],\
                    [6.4, 8.2, 3.9, 8.8],\
                    [5.0, 10.3, 3.6, 7.2],\
                    [3.8, 9.8, 4.4, 8.8],\
                    [3.3, 7.6, 4.3, 8.5]]
    for i in range(11):
        for j in range(4):
            observations[i][j] = int(observations[i][j] * 10)

class hmm(object):
    #A: Transition matrix ()
    #B: emission probabiltiy
    #pie: initial probability
    def __init__(self, A, B, pie):
        self.A = A
        self.B = B
        self.pie = pie
    def predict(self, obs):
        self.delta = np.zeros([len(obs), self.A.shape[0]])
        self.phi = np.zeros([len(obs), self.A.shape[0]])
        for k in range(len(obs)):
            for i in range(self.A.shape[0]):
                bi = 1
                for d, evidence in enumerate(obs[k]):
                    bi *= self.B[d][evidence][i]
                
                if k == 0:
                    self.delta[k][i] = self.pie[i] * bi
                else:
                    maxDelta = -float("Inf")
                    maxIndex = -1
                    for j in range(self.A.shape[0]):
                        delta = self.delta[k - 1][j] * \
                        self.A[j][i] * bi
                        if delta > maxDelta:
                            maxDelta = delta
                            maxIndex = j
                    self.delta[k][i] = maxDelta
                    self.phi[k][i] = maxIndex
        last = int(np.argmax(self.delta[-1]))
        res = []
        for i in range(len(obs) - 1, -1, -1):
            res.append(last)
            last = int(self.phi[i][last])
        return res[::-1]

if __name__ == "__main__":
    hmmMod = hmm(trans_prob, emission_prob, init_prob)
    res = hmmMod.predict(observations)
    for i in range(len(res)):
        temp = str(decoder[res[i]])
        res[i] = (int(temp[0]), int(temp[1]))
    g = graph.tolist()
    for x, y in res:
        g[x][y] = 9
    print("The most likely trace is: ")
    print(res)
    print("The 9 is the grid wherer the robot is in in each time step")
    for i in range(10):
        print(g[i])