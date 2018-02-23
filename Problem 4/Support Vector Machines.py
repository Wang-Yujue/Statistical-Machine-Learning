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

def gaussianKernel(x1, x2, sigma):
    x1 = x1[:]
    x2 = x2[:]
    sim = np.exp(-sum((x1 - x2) ^ 2) / (2 * sigma ^ 2))
    return sim

data = read_dataset('dataSets/iris-pca.txt')
label = data[:,2]
feature = data[:,[0,1]]

setosa_x1 = []
setosa_x2 = []
virginica_x1 = []
virginica_x2 = []
j = -1
for i in label:
    i = int(i)
    j = j + 1
    if i == 0:
        x1 = data[j,0]
        x2 = data[j,1]
        setosa_x1.append(x1)
        setosa_x2.append(x2)
    if i == 2:
        x1 = data[j, 0]
        x2 = data[j, 1]
        virginica_x1.append(x1)
        virginica_x2.append(x2)

plt.plot(setosa_x1,setosa_x2,'+')
plt.plot(virginica_x1,virginica_x2,'o')
plt.show()