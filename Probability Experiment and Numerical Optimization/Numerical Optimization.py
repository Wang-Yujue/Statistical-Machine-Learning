import numpy as np
import math
import matplotlib.pyplot as plt

# define rosenbrock function
def Rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def d_Rosenbrock(x):
    xm = x[1:-1]        # xm
    xm_m1 = x[:-2]      # xm_minus_1
    xm_p1 = x[2:]       # xm_plus_1
    d_r = np.zeros_like(x)
    d_r[1:-1] = np.multiply(200, (xm - xm_m1**2)) - \
                400 * np.multiply(xm_p1 - xm**2, xm) - 2 * (1 - xm)
    d_r[0] = -400 * np.multiply(x[0],(x[1]-x[0]**2)) - 2 * (1 - x[0])
    d_r[-1] = 200 * (x[-1] - x[-2]**2)
    return d_r

# define finite gradient descent function
def num_gradient_descent(learning_rate, x, num_iteration):
    rosen = []
    for i in range(num_iteration):
        gradient = d_Rosenbrock(x)
        x -= learning_rate * gradient
        rosen.append(Rosenbrock(x))
    return rosen

num_iteration = 100
x = np.random.random_sample((20,1))
learning_rate = 0.001
step = np.arange(num_iteration)
out = num_gradient_descent(learning_rate,x,num_iteration)

# plot figure
plt.figure()
plt.plot(step, out)
plt.xlabel('Steps')
plt.ylabel('Rosenbrock Value')
plt.show()