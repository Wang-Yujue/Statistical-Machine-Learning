import numpy as np
from math import log
from matplotlib import pyplot as plt


# 2.3.a
bin_size = 0.02

def read(path):
    data = []
    txt = open(path)
    for line in txt:
        data.append(float(line.strip()))
    return np.asarray(data)

train = read("dataSets/nonParamTrain.txt")
test = read("dataSets/nonParamTest.txt")

bins = int((max(train)-min(train))/bin_size+1)
plt.hist(train,bins=bins,edgecolor = 'black')
plt.title('bins size 0.02')
plt.show()


# 2.3.b
def kde(sigma,x1,x2):
    f = []
    for x0 in x2:
        sum = 0
        for xn in x1:
            u = xn - x0
            K = np.exp(- u**2 /(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
            sum += 1/(len(x1)) * K
        f.append(sum)
    return f

def log_likelihood(f):
    loss = [log(fi) for fi in f]
    return sum(loss)


x_kde = sorted(train)
f1 = kde(0.03,train,x_kde)
f2 = kde(0.2,train,x_kde)
f3 = kde(0.8,train,x_kde)


plt.figure()
plt.plot(x_kde,f1,'b',label='sigma = 0.03')
plt.plot(x_kde,f2,'r',label='sigma = 0.2')
plt.plot(x_kde,f3,'g',label='sigma = 0.8')
plt.xlim([-4,8])
plt.legend()
plt.show()

# print(log_likelihood(f1),log_likelihood(f2),log_likelihood(f3))


# 2.3.c
def knn(data,x,k):
    result = []
    for xi in x:
        dist = [np.abs(point-xi) for point in data]
        V = np.sort(dist)[k]
        p = k/(V*len(data))
        result.append(p)
    return result

x_knn = np.linspace(-4,8,200)
knn_1 = knn(train,x_knn,2)
knn_2 = knn(train,x_knn,8)
knn_3 = knn(train,x_knn,35)


plt.figure()
plt.plot(x_knn,knn_1,'b',label='k=2')
plt.plot(x_knn,knn_2,'r',label='k=8')
plt.plot(x_knn,knn_3,'g',label='k=35')
plt.xlim([-4,8])
plt.show()

test = sorted(test)

# 2.3.d
test_kde_1 = kde(0.03,train,test)
test_kde_2 = kde(0.2,train,test)
test_kde_3 = kde(0.8,train,test)

test_knn_1 = knn(train,test,2)
test_knn_2 = knn(train,test,8)
test_knn_3 = knn(train,test,35)

print('test-set: kernel density estimation:',log_likelihood(test_kde_2),log_likelihood(test_kde_3))
print('test-set: k nearest neighbor:',log_likelihood(test_knn_1),log_likelihood(test_knn_2),log_likelihood(test_knn_3))

print('train-set: kernel density estimation:',log_likelihood(f1),log_likelihood(f2),log_likelihood(f3))
print('train-set: k nearest neighbor:',log_likelihood(knn_1),log_likelihood(knn_2),log_likelihood(knn_3))

