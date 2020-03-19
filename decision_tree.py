import numpy as np
import pandas as pd
from math import log2
import graphviz


class Dtree(object):
    ###This is the main frame of the tree
    def __init__(self, data: pd.DataFrame, questions: set, labelName: str):
        ###The data of each split
        self.data = data
        ###questions to be asked
        self.questions = questions
        ###The question that yield the most information gain
        self.bestQuestion = None
        ###Entropy of the data
        self.entropy = None
        ###The amount of entropy reduced after the data splitted by questions
        self.entropyReduction = None
        ###Name of the label column
        self.labelName = labelName
        ###Next node of the decision tree
        self.yes = None
        self.no = None
    
    
    ###A function that calculates entropy for data
    def getEntropy(self, labels: pd.DataFrame):
        counts = labels.value_counts()
        total = len(counts)
        population = len(labels)
        entropy = 0
        for i in range(total - 1):
            entropy += - ((counts[i]/population)*log2(counts[i]/population) + 
                          ((population - counts[i])/population)*
                          log2((population - counts[i])/population))
        return entropy
    
    
    
    
    ###A function that splits the data by answering the question
    def makeDecision(self, question: tuple, data: pd.DataFrame):
        yesSplit = []
        noSplit = []
        for i in range(len(data[question[0]])):
            if data[question[0]].iloc[i] == question[1]:
                yesSplit.append(data.iloc[i])
            else:
                noSplit.append(data.iloc[i])
        
        return pd.DataFrame(yesSplit, columns = data.columns), \
    pd.DataFrame(noSplit,columns = data.columns)
    
    
    
    
    ###Calculate the information gain by asking each question,
    ####choose the question that yields th
    def infoGain(self, question: tuple, data: pd.DataFrame, labelName: str):
        yesSplit, noSplit = self.makeDecision(question, data)
        ratio = len(yesSplit[labelName])/len(data[labelName])
        gain = self.getEntropy(data[labelName]) - \
        (ratio * self.getEntropy(yesSplit[labelName]) + \
         (1 - ratio) * self.getEntropy(noSplit[labelName]))
        return gain, yesSplit, noSplit
    
    
    
    ###A function that trains the decision tree recursively
    def train(self):
        
        self.entropy = self.getEntropy(self.data[self.labelName])
        
        if not self.questions or self.entropy <= 0.02:
            return
        
        maxGain = 0
        maxYesSplit = None
        maxNoSplit = None
        bestQuestion = None
        
        for question in questions:
            gain, yesSplit, noSplit = self.infoGain(question, 
                                                    self.data, 
                                                    self.labelName)
            if gain > maxGain:
                maxGain = gain
                maxYesSplit = yesSplit
                maxNoSplit = noSplit
                bestQuestion = question
        
        self.bestQuestion = bestQuestion
        self.entropyReduction = maxGain
        
        quests = self.questions.copy()
        if maxGain >= 0.02:
            
            quests.remove(bestQuestion)

            if not maxYesSplit.empty:

                self.yes = Dtree(maxYesSplit, quests, self.labelName)
                self.yes.train()

            if not maxNoSplit.empty:
                self.no = Dtree(maxNoSplit, self.questions.copy(), 
                                self.labelName)
                self.no.train()
        else:
            return
        
    
    ###A function that makes prediction after training
    def predict(self, inputAttribute: set):
        if not self.yes and not self.no:
            print(self.data[self.labelName].value_counts().idxmax())
            return self.data[self.labelName].value_counts().idxmax()
        if self.bestQuestion in inputAttribute:
            self.yes.predict(inputAttribute)
        else:
            self.no.predict(inputAttribute)
    
    
    ###Tree traversal to get the structure of tree
    def BFS(self):
        queue = [self]
        output = []
        graph = []
        while queue:
            temp = []
             
            for node in queue:
                if node.yes is None and node.no is None:
                    output.append(node)
                graph.append(node)
                if node.yes is not None:
                    temp.append(node.yes)
                if node.no is not None:
                    temp.append(node.no)
            
            queue = temp
        
        return output, graph
    
    
    ###Generate the label for graph
    def graphvizHTML(self):
        if self.yes and self.no:
            
            return "<" + str(self.bestQuestion) + \
        r'<BR /><FONT POINT-SIZE="10">'+"Entropy: "+ \
        str(self.entropy)+'</FONT>'+ \
        r'<BR /><FONT POINT-SIZE="10">'+ \
        'Max information gain: ' + \
        str(self.entropyReduction) + '</FONT>'+'>'
        
        else:
            return "<" + 'Class: ' + \
        str(self.data[self.labelName].value_counts().idxmax()) + \
        r'<BR /><FONT POINT-SIZE="10">'+"Entropy: "+ \
        str(self.entropy)+'</FONT>'+ \
        r'<BR /><FONT POINT-SIZE="10">'+ \
        'Max information gain: ' + \
        str(self.entropyReduction) + '</FONT>'+'>'
    
    
    ###Tree traversal to get the structure of tree
    ### dot is a graphviz object
    def printTree(self,dot):
        if self.yes and self.no:
            dot.node(str(id(self.yes)), self.yes.graphvizHTML())
            dot.node(str(id(self.no)), self.no.graphvizHTML())
            dot.edge(str(id(self)),str(id(self.yes)),'yes')
            dot.edge(str(id(self)),str(id(self.no)),'no')
            self.yes.printTree(dot)
            self.no.printTree(dot)