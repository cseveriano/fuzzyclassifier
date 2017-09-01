import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split

import FuzzySystem
import Partitioner as part


# import base iris
iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target


#for npaticoes in range(2,6):
#    for iteracoes in range(1,10):
# normalizar X e y
nparticoes = 2
X_norm = X / X.max(axis=0)


# separar base com Cross Validation
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=0)

fuzzysets = []

for i in range(X_train.shape[1]):
    fuzzysets.append(part.Partitioner("FS", np.array([0,1]), nparticoes).sets)

fuzzy = FuzzySystem.FuzzySystem(fuzzysets)
fuzzy.train(X_train, y_train)
error = fuzzy.test(X_test, y_test)


