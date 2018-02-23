import numpy as np
import matplotlib.pyplot as plt

data_dim = 5

# read data and normalize
def read_dataset(path):
    data = []
    txt = open(path)
    for line in txt:
        a,b,c,d,e = map(float,line.strip().split(","))
        data.append((a,b,c,d,e))
    return np.asarray(data)

def normalize(data):
    return (data - data.mean(0)) / np.std(data,0)

# calculate eigenvalues & eigenvectors and visualize them
def eigendecomposition(train):
    vals,vecs = np.linalg.eig(np.cov(train.T))
    sort_idx = np.argsort(vals)[::-1]
    vals_sort = vals[sort_idx]
    vecs_sort = vecs[:,sort_idx]
    return vals_sort,vecs_sort

def propotion(vals):
    cum_var = np.cumsum(vals)
    var_pro = (cum_var / cum_var[-1]) * 100
    return var_pro

# low dimensional representation
def representation(train,vecs,dim):
    cord = np.zeros((len(train),dim))
    for i,x in enumerate(train):
        cord[i,:] = vecs[:,:dim].T.dot(x)
    return cord


data = read_dataset('dataSets/iris.txt')
train = data[:,:4]

label = data[:,-1]
N = len(label)

norm_train = normalize(train)
vals,vecs = eigendecomposition(norm_train)

var_pro = propotion(vals)
plt.plot(range(1,5),var_pro)
plt.show()

x_cord = representation(train,vecs,dim=2)
color = ['r','g','b']
plt.scatter(x_cord[:,0],x_cord[:,1],c=[color[int(i)] for i in label])
plt.show()

# projection to original space
for i in range(1,5):
    x_cordn = representation(train,vecs,dim = i)
    x_proj= np.zeros((N,4))
    for n, x in enumerate(x_cordn):
        x_proj[n,:] = vecs[:,:i].dot(x)
    error = [np.sqrt(sum((x_proj[:,m] - train[:,m])**2)/N) for m in range(4)]
    print(error)