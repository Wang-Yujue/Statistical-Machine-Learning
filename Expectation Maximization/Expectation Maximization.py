import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

filename = 'dataSets/gmm.txt'
x_file = []
y_file = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
        x_tmp, y_tmp = [float(i) for i in lines.split()]
        x_file.append(x_tmp)
        y_file.append(y_tmp)

x_file = np.array(x_file)
y_file = np.array(y_file)


# Initialize with parameters roughly getting from the plot
mu_1 = [0,0]
cov_1 = np.array([[2, 0], [0, 1]])
pi_1 = 1/4
mu_2 = [0,2]
cov_2 = np.array([[0.5, 0], [0, 0.5]])
pi_2 = 1/4
mu_3 = [2,4]
cov_3 = np.array([[1.5, 0], [0, 1.5]])
pi_3 = 1/4
mu_4 = [2,1]
cov_4 = np.array([[1, 0], [0, 1]])
pi_4 = 1/4


def gaussian_2d(x, y, mu, cov):
    return mlab.bivariate_normal(x,y,np.sqrt(cov[0,0]),np.sqrt(cov[1,1]),mu[0],mu[1],np.sqrt(cov[0,1]))

for i in range(5):
    i = i + 1

    gaussian_1 = gaussian_2d(x_file,y_file, mu_1, cov_1)
    gaussian_2 = gaussian_2d(x_file,y_file, mu_2, cov_2)
    gaussian_3 = gaussian_2d(x_file,y_file, mu_3, cov_3)
    gaussian_4 = gaussian_2d(x_file,y_file, mu_4, cov_4)

    p_y = pi_1 * gaussian_1 + pi_2 * gaussian_2 + pi_3 * gaussian_3 + pi_4 * gaussian_4
    q1 = (pi_1 * gaussian_1) / p_y
    q2 = (pi_2 * gaussian_2) / p_y
    q3 = (pi_3 * gaussian_3) / p_y
    q4 = (pi_4 * gaussian_4) / p_y

    mu_1 = [np.sum(q1 * x_file), np.sum(q1 * y_file)] / np.sum(q1)
    # sigxy1 = np.sum(q1*(x_file - mu_1[0])* ((y_file - mu_1[1])))
    sigxy1 = 0
    cov_1 = [[np.sum(q1 * ((x_file - mu_1[0]) ** 2)), sigxy1], [sigxy1, np.sum(q1 * ((y_file - mu_1[1]) ** 2))]] / np.sum(q1)
    pi_1 = np.sum(q1) / 500
    # sigxy2 = np.sum(q1*(x_file - mu_2[0])* ((y_file - mu_2[1])))
    sigxy2 = 0
    mu_2 = [np.sum(q2 * x_file), np.sum(q2 * y_file)] / np.sum(q2)
    cov_2 = [[np.sum(q2 * ((x_file - mu_2[0]) ** 2)), sigxy2], [sigxy2, np.sum(q2 * ((y_file - mu_2[1]) ** 2))]] / np.sum(q2)
    pi_2 = np.sum(q2) / 500
    # sigxy3 = np.sum(q1 * (x_file - mu_3[0]) * ((y_file - mu_3[1])))
    sigxy3 = 0
    mu_3 = [np.sum(q3 * x_file), np.sum(q3 * y_file)] / np.sum(q3)
    cov_3 = [[np.sum(q3 * ((x_file - mu_3[0]) ** 2)), sigxy3], [sigxy3, np.sum(q3 * ((y_file - mu_3[1]) ** 2))]] / np.sum(q3)
    pi_3 = np.sum(q3) / 500
    # sigxy4 = np.sum(q1*(x_file - mu_4[0])* ((y_file - mu_4[1])))
    sigxy4 = 0
    mu_4 = [np.sum(q4 * x_file), np.sum(q4 * y_file)] / np.sum(q4)
    cov_4 = [[np.sum(q4 * ((x_file - mu_4[0]) ** 2)), sigxy4], [sigxy4, np.sum(q4 * ((y_file - mu_4[1]) ** 2))]] / np.sum(q4)
    pi_4 = np.sum(q4) / 500


x = np.linspace(-2,4)
y = np.linspace(-2,4)
X,Y = np.meshgrid(x, y)

# plot contour result
plt.figure()
x1,y1 = np.mgrid[-2:4:.01,-2:4:.01]
pos = np.empty(x1.shape + (2,))
pos[:, :, 0] = x1; pos[:, :, 1] = y1
z1= multivariate_normal(mu_1,cov_1)

x2,y2 = np.mgrid[-2:4:.01,-2:4:.01]
pos2 = np.empty(x2.shape + (2,))
pos2[:, :, 0] = x2; pos2[:, :, 1] = y2
z2= multivariate_normal(mu_2,cov_2)

x3,y3 = np.mgrid[-2:4:.01,-2:4:.01]
pos3 = np.empty(x3.shape + (2,))
pos3[:, :, 0] = x3; pos3[:, :, 1] = y3
z3= multivariate_normal(mu_3,cov_3)


x4,y4 = np.mgrid[-2:4:.01,-2:4:.01]
pos4 = np.empty(x4.shape + (2,))
pos4[:, :, 0] = x4; pos4[:, :, 1] = y4
z4= multivariate_normal(mu_4,cov_4)

c1=plt.scatter(x_file,y_file,s=20,alpha=.5)
plt.contour(x1,y1,z1.pdf(pos))
plt.contour(x2,y2,z2.pdf(pos2))
plt.contour(x3,y3,z3.pdf(pos3))
plt.contour(x4,y4,z4.pdf(pos4))
plt.title('Data points and the probability densities of each class')
plt.show()
