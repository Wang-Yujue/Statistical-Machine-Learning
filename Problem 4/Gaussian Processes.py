import numpy as np
import matplotlib.pyplot as plt


def SE_Kernel(t1 ,t2, sigma_f, l):
    K_ff = sigma_f ** 2 * np.exp(-1 / (2 * l ** 2) * (t1 - t2) ** 2)
    return K_ff

def Delta(t1, t2):
    if t1 == t2:
        d = 1
    else:
        d = 0
    return d

def Noise_Kernel(t1 ,t2, sigma_n):
    K_nn = sigma_n ** 2 * Delta(t1, t2)
    return K_nn

x = np.arange(0.0, 2.0 * np.pi, 0.005)
y = np.sin(x) + np.sin(x)**2

sigma_f = 1 # set appropriate std according to the ground truth
l = 0.5 # choose proper hyperparameter to avoid overfitting and underfitting
sigma_n = np.sqrt(0.001) # noise variance

# first plot prior
N = len(x)
K_ff = np.zeros([N, N])
K_nn = np.zeros([N, N])
mean = np.zeros(N)

for i in range(0, N):
    for j in range(0, N):
        K_ff[i, j] = SE_Kernel(x[i], x[j], sigma_f, l)
        K_nn[i, j] = Noise_Kernel(x[i], x[j], sigma_n)

f = np.random.multivariate_normal(mean, K_ff)
#f = np.transpose(np.linalg.cholesky(K_ff)) * np.random.normal(0,1,N)
plt.figure()
plt.plot(x, f)

K_yy = K_ff + K_nn
K_fy = np.zeros([N, 1])
sigma2 = K_ff # initial value
#mu = mean # initial value
for ite in range(0, 1):
    s_p = np.argmax(np.abs(f))
    x_s = x[s_p] # sample point
    y_s = np.sin(x_s) + np.sin(x_s)**2 # sample
    for i in range(0, N):
        K_fy[i] = SE_Kernel(x[i], x_s, sigma_f, l)

    mu = np.dot(np.transpose(K_fy), np.linalg.inv(K_yy)) * (y_s - np.mean(y))
    #mu = np.dot(np.transpose(K_fy), np.linalg.inv(K_yy)) * (y_s - 0)
    #mu = np.dot(np.transpose(K_fy), np.linalg.inv(K_yy))
    mu = np.reshape(mu, N)
    #mu[s_p] = mu[s_p] + y_s
    sigma2[s_p,:] = K_ff[:,s_p] - np.dot(np.dot(np.transpose(K_fy), np.linalg.inv(K_yy)), K_fy)

    f = np.random.multivariate_normal(mu, sigma2)
    plt.plot(x_s, y_s, 'o')
    plt.plot(x, f) # prediction
    #plt.figure()
    ite = ite + 1

std = np.zeros(N)
for j in range(0, N):
    std[j] = np.sqrt(sigma2[j, j])

plt.plot(x, mu + 2 * std)
plt.plot(x, mu - 2 * std)
plt.plot(x, y)
plt.plot(x, mu)

plt.show()