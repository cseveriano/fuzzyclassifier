import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split

import FuzzySystem
import Partitioner as part


# import base iris
iris = datasets.load_iris()

#X = iris.data[:,:2]
X = iris.data
y = iris.target

# normalizar X
X_norm = X / X.max(axis=0)

testes_particoes = np.arange(2,11)

train_errors = []
test_errors = []

for nparticoes in testes_particoes:

    # separar base com Cross Validation
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.5, random_state=0)

    fuzzysets = []

    for i in range(X_train.shape[1]):
        fuzzysets.append(part.Partitioner("FS", np.array([0,1]), nparticoes).sets)

    fuzzy = FuzzySystem.FuzzySystem("FuzzyWithCG", fuzzysets)
    fuzzy.train(X_train, y_train)

    train_errors.append(fuzzy.test(X_train, y_train))
    test_errors.append(fuzzy.test(X_test, y_test))


for part, train, test in zip(testes_particoes,train_errors,test_errors):
    print("Particoes = ",part)
    print("Erro treinamento = ",train)
    print("Erro teste = ",test)
