import numpy as np
import matplotlib.pyplot as plt

# load and separate data
data_all = np.loadtxt('dataSets/ldaData.txt')
data_C1 = data_all[:50,:]
data_C2 = data_all[50:100,:]
data_C3 = data_all[100:,:]

ones1 = np.ones((np.shape(data_C1)[0],1))
ones2 = np.ones((np.shape(data_C2)[0],1))
ones3 = np.ones((np.shape(data_C3)[0],1))
zeros1 = np.zeros((np.shape(data_C1)[0],1))
zeros2 = np.zeros((np.shape(data_C2)[0],1))
zeros3 = np.zeros((np.shape(data_C3)[0],1))
points_C1_1 = np.concatenate( [ones1,data_C1], 1 )
points_C2_1 = np.concatenate( [ones2,data_C2], 1 )
points_C3_1 = np.concatenate( [ones3,data_C3], 1 )
points_all_1 = np.concatenate( [points_C1_1,points_C2_1,points_C3_1], 0 )
labels_C1 = np.concatenate( [ones1,zeros1,zeros1], 1 )
labels_C2 = np.concatenate( [zeros2,ones2,zeros2], 1 )
labels_C3 = np.concatenate( [zeros3,zeros3,ones3], 1 )
labels_all = np.concatenate( [labels_C1,labels_C2,labels_C3], 0 )

# using least squares
X = points_all_1
T = labels_all
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(T)

Predictions = np.zeros(np.shape(labels_all))
for i in range(np.shape(Predictions)[0]):
    Predictions[i,:] = W.T.dot(X[i,:].T)
predictions = np.argmax(Predictions,1)+1
points_pred_C1 = data_all[predictions == 1,:]
points_pred_C2 = data_all[predictions == 2,:]
points_pred_C3 = data_all[predictions == 3,:]

truth = np.argmax(labels_all,1)+1
Error = np.zeros(np.shape(truth))
Error[truth != predictions] = 1
num_misclassified = np.sum(Error)
print(num_misclassified)

plt.figure()
h_C1 = plt.plot(points_C1_1[:,1],points_C1_1[:,2], 'x')
h_C2 = plt.plot(points_C2_1[:,1],points_C2_1[:,2], 'o')
h_C3 = plt.plot(points_C3_1[:,1],points_C3_1[:,2], '*')
plt.show()

plt.figure()
h_C1p = plt.plot(points_pred_C1[:,0],points_pred_C1[:,1], 'x')
h_C2p = plt.plot(points_pred_C2[:,0],points_pred_C2[:,1], 'o')
h_C3p = plt.plot(points_pred_C3[:,0],points_pred_C3[:,1], '*')
plt.show()