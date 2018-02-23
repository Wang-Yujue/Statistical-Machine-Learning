import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

def read(path):
    data = []
    txt = open(path)
    for line in txt:
        x,y=map(float,line.strip().split())
        data.append((x,y))
    return np.asarray(data)

c1 = read("dataSets/densEst1.txt")
c2 = read("dataSets/densEst2.txt")

# calculate prior
prior_c1 = len(c1)/(len(c1)+len(c2))
prior_c2 = len(c2)/(len(c1)+len(c2))

# calculate mu and sigma
def estimate(data,biased=True):
    mu = np.mean(data,0)
    N = len(data)
    norm = N if biased else N-1
    sigma = 1/norm * sum((x-mu).reshape(2,1)*(x-mu) for x in data)
    return mu, sigma

mu_c1, sigma_biased_c1 = estimate(c1,biased=True)
_, sigma_unbiased_c1 = estimate(c1,biased=False)

mu_c2, sigma_biased_c2 = estimate(c2,biased=True)
_, sigma_unbiased_c2 = estimate(c2,biased=False)

print('Class 1: mu:',mu_c1,'Biased sigma:',sigma_biased_c1,'Unbiased sigma',sigma_unbiased_c1)
print('Class 2: mu:',mu_c2,'Biased sigma:',sigma_biased_c2,'Unbiased sigma',sigma_unbiased_c2)

# plot contour result
plt.figure()
x1,y1 = np.mgrid[-10:10:.01,-10:10:.01]
pos = np.empty(x1.shape + (2,))
pos[:, :, 0] = x1; pos[:, :, 1] = y1
z1= multivariate_normal(mu_c1,sigma_unbiased_c1)

x2,y2 = np.mgrid[-10:12.5:.01,-10:12.5:.01]
pos2 = np.empty(x2.shape + (2,))
pos2[:, :, 0] = x2; pos2[:, :, 1] = y2
z2= multivariate_normal(mu_c2,sigma_unbiased_c2)

c1=plt.scatter(c1[:,0],c1[:,-1],color='red',s=20,alpha=.5)
c2=plt.scatter(c2[:,0],c2[:,-1],color='blue',s=20,alpha=.4)
plt.legend((c1,c2),('class 1','class 2'))
plt.contour(x1,y1,z1.pdf(pos))
plt.contour(x2,y2,z2.pdf(pos2))
plt.title('Data points and the probability densities of each class')
plt.show()

# plot posterior result
# check duplicates here
# dup1 = [x for n,x in enumerate(c1) if x in c1[:n]]
# dup2 = [x for n,x in enumerate(c2) if x in c2[:n]]
# dup_cross = [x for n,x in enumerate(c1) if x in c2]
# print('no duplicates') if dup1==[]  else print(dup1,dup2,dup_cross)
# # it turns out there's no duplicates in dataset
#
# def calculate_posterior(c1,c2):
#     likelihood_c1 = c1/len(c1)
#     likelihood_c2 = c2/len(c2)
#     posterior_c1 = likelihood_c1 * prior_c1/(1/(len(c1)+len(c2)))
#     posterior_c2 = likelihood_c2 * prior_c2/(1/(len(c1)+len(c2)))
#     return np.asarray(posterior_c1),np.asarray(posterior_c2)
#
#
# plt.figure()
# posterior_c1,posterior_c2 = calculate_posterior(c1,c2)


