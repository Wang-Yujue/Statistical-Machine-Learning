import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# read data
def read_dataset(path):
    data = []
    txt = open(path)
    for line in txt:
        a,b,c = map(float,line.split())
        data.append([a,b,c])
    return np.asarray(data)

def gaussianKernel(x1 ,x2):
    sigma = 2
    sim = np.exp(-sum(np.subtract(x1, x2) ** 2) / (2 * sigma ** 2))
    return sim

data = read_dataset('dataSets/iris-pca.txt')
Y = data[:,2]
X = data[:,[0,1]]
s = 0.02 # mesh step size
fignum = 1

# fit the model
for kernel in ('linear', 'rbf'):
    clf = svm.SVC(kernel=kernel, C = 1)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, s), np.arange(y_min, y_max, s))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(fignum)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    fignum = fignum + 1

plt.show()