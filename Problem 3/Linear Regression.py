import numpy as np
import matplotlib.pyplot as plt

# a)
def get_polynomial_feature_matrix(points, num_polynomials):
    num_points = len(points)
    feature_matrix = np.zeros((num_points,num_polynomials))
    for i in range(num_points):
        for j in range(num_polynomials):
            feature_matrix[i,j] = points[i] ** j
    return feature_matrix

def ridge_linear_regression(data, feature_matrix, ridge_coeff):
    labels = data[ :,1]
    X = feature_matrix
    num_features = np.shape(X)[1]
    preInv = (X.T).dot(X) + np.eye(num_features) * ridge_coeff
    theta = np.linalg.inv(preInv).dot(X.T).dot(labels)
    return theta

def calc_rmse(true, prediction):
    num_points = np.shape(true)[0]
    err = true-prediction
    rmse = np.sqrt((err.T).dot(err)/num_points)
    return rmse

def predict_polynomial(points, theta):
    num_polynomials = np.size(theta)
    X = get_polynomial_feature_matrix(points, num_polynomials)
    prediction = X.dot(theta)
    return prediction

data_all = np.loadtxt('dataSets/linRegData.txt')
data_train = data_all[:20,:]
data_eval = data_all[20:,:]
points_train = data_train[:,0]
points_eval = data_eval[:,0]
true_train = data_train[:,1]
true_eval = data_eval[:,1]

max_polynomial = 21
ridge_coef = 10**(-6)

train_err = []
eval_err = []
for num_features in range(1,max_polynomial+1):
    feature_matrix = get_polynomial_feature_matrix(points_train, num_features)
    theta = ridge_linear_regression(data_train, feature_matrix, ridge_coef)
    predictions_train = predict_polynomial(points_train,theta)
    predictions_eval = predict_polynomial(points_eval,theta)
    train_err.append(calc_rmse(true_train, predictions_train))
    eval_err.append(calc_rmse(true_eval, predictions_eval))

num_poly = np.argmin(eval_err)
print(num_poly)
best_feature_matrix = get_polynomial_feature_matrix(points_train, np.argmin(eval_err))
theta_best = ridge_linear_regression(data_train,best_feature_matrix,ridge_coef)
model_input = np.arange(0,2,0.01)
best_model = predict_polynomial(model_input,theta_best)

plt.figure()
h_all_data = plt.scatter(data_all[:,0],data_all[:,1],label="Ground Truth")
h_model, = plt.plot(model_input,best_model,label="Prediction")
plt.legend(handles = [h_all_data, h_model])
plt.show()

plt.figure()
h_train_err, = plt.plot(train_err, label="Training Data")
h_eval_err, = plt.plot(eval_err, label="Testing Data")
plt.legend(handles = [h_train_err,h_eval_err])
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.show()

# b)
def eval_gaus(x, mu, sig2):
    exp = np.exp(-(x - mu)**2 / (2*sig2))
    return exp

def get_gaussian_feature_matrix(points, num_centers):
    sig2 = 0.02
    dist = 2.0/(num_centers-1)
    mu = np.arange(0,2.001,dist)
    feature_matrix = np.zeros( (len(points),len(mu)) )
    for i in range(len(mu)):
        feature_matrix[:,i] = eval_gaus(points, mu[i], sig2)
    return feature_matrix

def predict_rbf(points, theta):
    num_centers = np.size(theta)
    X = get_gaussian_feature_matrix(points, num_centers)
    prediction = X.dot(theta)
    return prediction

numfeature = 20
feature_matrix = get_gaussian_feature_matrix(points_train,numfeature)
theta = ridge_linear_regression(data_train,feature_matrix,ridge_coef)
features = feature_matrix = get_gaussian_feature_matrix(model_input,numfeature)

# feature normalization and plot
plt.figure()
# three different normalization method leads slightly different style of this gaussian feature plot
for i in range(numfeature):
    # Rescaling
    features[:,i] = (features[:,i] - np.min(features[:,i])) / (np.max(features[:,i]) - np.min(features[:,i]))
    plt.plot(model_input, features[:, i])
plt.figure()
for i in range(numfeature):
    # Scaling to unit length
    features[:, i] = features[:, i] / np.linalg.norm(features[:, i])
    plt.plot(model_input, features[:, i])
plt.figure()
for i in range(numfeature):
    # Standardization
    features[:, i] = (features[:, i] - np.min(features[:, i])) / np.std(features[:, i])
    plt.plot(model_input, features[:, i])

plt.show()

# c)
train_err = []
eval_err = []
min_num_centers = 17
max_num_centers = 40
for num_cent in range(min_num_centers,max_num_centers+1):
    feature_matrix = get_gaussian_feature_matrix(points_train, num_cent)
    theta = ridge_linear_regression(data_train, feature_matrix, ridge_coef)
    predictions_train = predict_rbf(points_train,theta)
    predictions_eval = predict_rbf(points_eval,theta)
    train_err.append(calc_rmse(true_train, predictions_train))
    eval_err.append(calc_rmse(true_eval, predictions_eval))

plt.figure()
h_eval_err, = plt.plot(np.arange(min_num_centers,max_num_centers+1),eval_err, label="Testing Data")
plt.legend(handles = [h_eval_err])
plt.xlabel("Number of Basis Functions")
plt.ylabel("RMSE")
plt.show()

# d)
pol_rank = 12
num_train_samples = [10, 12, 16, 20, 50, 150]
model = predict_rbf(model_input, theta)

for i in range(len(num_train_samples)):
    # calculate model predictive mean
    data_train = data_all[:num_train_samples[i],:]
    points_train = data_train[:,0]
    true_train = data_train[:,1]
    feature_matrix = get_polynomial_feature_matrix(points_train, pol_rank)
    theta = ridge_linear_regression(data_train, feature_matrix, ridge_coef)
    predictions_train = predict_polynomial(points_train,theta)

    centered = true_train - predictions_train
    sig2model = (centered.T).dot(centered)/len(true_train)
    A = feature_matrix
    lambdaI = np.eye(np.shape(A)[1]) * ridge_coef
    inv = np.linalg.inv( (A.T).dot(A) + lambdaI )
    sig = []
    model_input = np.arange(0,2,0.01)
    for j in range(len(model_input)):
        x = get_polynomial_feature_matrix([model_input[j]],pol_rank).T
        sig.append( np.sqrt( (sig2model + sig2model * (x.T).dot(inv).dot(x))[0,0] )  )

    plt.figure()
    h_train_data = plt.scatter(data_train[:, 0], data_train[:, 1], label="train data")
    h_model, = plt.plot(model_input,model,label='Mean')
    h_std, = plt.plot(model_input,model+sig,color='black',linewidth=1.0, label = "Standard Deviation")
    plt.plot(model_input,model-sig,color='black',linewidth=1.0)
    plt.show()