import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split

import FuzzySystem
import Partitioner as part


# import base iris
iris = datasets.load_iris()

X = iris.data[:,:2]
#X = iris.data
y = iris.target

# normalizar X
X_norm = X / X.max(axis=0)

testes_particoes = np.arange(2,11)

train_errors = []
test_errors = []

# separar base com Cross Validation
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.5, random_state=0)

fuzzysets = []
nparticoes = 4

for i in range(X_train.shape[1]):
    fuzzysets.append(part.Partitioner("FS", np.array([0,1]), nparticoes).sets)

x = np.arange(0,1,0.01)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 9))

for sets in fuzzysets[0] :
    fs_y1 = [sets.membership(xx) for xx in x]
    ax1.plot(x, fs_y1, linewidth=1.5)
    ax1.set_title('Funções Pertinência - Comprimento Sépala')
    ax1.legend()
    plt.grid()

for sets in fuzzysets[1] :
    fs_y2 = [sets.membership(xx) for xx in x]
    ax2.plot(x, fs_y2, linewidth=1.5)
    ax2.set_title('Funções Pertinência - Largura Sépala')
    ax2.legend()
    ax2.grid()

for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()


#plt.figure()
#for sets in fuzzysets[0] :
#    fs_y1 = [sets.membership(xx) for xx in x]
#    fs_y1 = [(a - 1)*0.1 for a in fs_y1]
#    plt.plot(x, (fs_y1), linewidth=1.5)

#for sets in fuzzysets[1] :
#    fs_y2 = [sets.membership(xx) for xx in x]
#    fs_y2 = [(a - 1)*0.1 for a in fs_y2]
#    plt.plot((fs_y2), x, linewidth=1.5)

#plt.show()

#ax.set_xticks(np.arange(0, 1, 0.1))
#ax.set_yticks(np.arange(0, 1., 0.1))

plt.figure()
fuzzy = FuzzySystem.FuzzySystem("FuzzyWithCG", fuzzysets)
fuzzy.train(X_train, y_train)

train_consequents, train_error_rate = fuzzy.test(X_train, y_train)

for i in range(X_train.shape[0]):
    x, y = [X_train[i][0], X_train[i][1]]
    color = ""
    output = train_consequents[i]

    if output == 0:
        color = "red"
    elif output == 1:
        color = "green"
    elif output == 2:
        color = "blue"

    plt.scatter(x, y, c=color, alpha=1, edgecolor="none")

# Legend
red_patch = mpatches.Patch(color='red', label='iris setosa')
green_patch = mpatches.Patch(color='green', label='iris versicolor')
blue_patch = mpatches.Patch(color='blue', label='iris virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])

x = np.arange(0,1,0.01)
for sets in fuzzysets[0] :
    fs_y1 = [sets.membership(xx) for xx in x]
    fs_y1 = [(a - 1)*0.1 for a in fs_y1]
    plt.plot(x, (fs_y1), linewidth=1.5)

for sets in fuzzysets[1] :
    fs_y2 = [sets.membership(xx) for xx in x]
    fs_y2 = [(a - 1)*0.1 for a in fs_y2]
    plt.plot((fs_y2), x, linewidth=1.5)

plt.title("Classificação Base Treinamento Iris", fontsize=18)
plt.xlabel(r'Comprimento Sépala', fontsize=15)
plt.ylabel(r'Largura Sépala', fontsize=15)

plt.legend()
plt.grid(True)
plt.show()
#fuzzy = FuzzySystem.FuzzySystem("FuzzyWithCG", fuzzysets)
#fuzzy.train(X_train, y_train)

