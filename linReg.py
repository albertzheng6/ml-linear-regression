import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import pandas as pd

# Get X from the initial 2d array of data
def getX(mat):
    X = np.array(mat[:, 0:mat.shape[1] - 1])
    return X

# Get y from the initial 2d array of data
def getY(mat):
    y = np.array([mat[:, mat.shape[1] - 1]])
    y = y.T  # T can only be called on numpy arrays, not arrays
    return y

# Defined hypothesis for linear regression
def h(x, theta):
    x = x.astype(float)
    return np.dot(x, theta)

# Precondition: X only has columns x_0 and x_1
def plotLinReg(X,y,theta_min):
    plt.scatter(X[:, 1], y, s=5)

    x_reg = np.linspace(4, 25, 100)
    y_reg = theta_min[0] + theta_min[1] * x_reg
    plt.plot(x_reg, y_reg, 'r')

    plt.style.use('seaborn-whitegrid')
    plt.xlabel('xi')
    plt.ylabel('y')
    plt.title('xi vs y')
    plt.xlim(4, 25)
    plt.ylim(-5, 24)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Computes cost function with theta_0 and theta_1 as input values
# Returns float value
def computeCost1(X, y, theta):
    m = len(y)
    sum = 0.0
    for i in range(0,m):
        hyp = theta[0,0] * X[i,0] + theta[1,0] * X[i,1]
        sum += (hyp - y[i,0])**2
    res = (1/(2*m)) * sum
    return round(res,15)

# Computes cost function with theta_0, theta_1, and theta_2 as input values
# Returns float value
def computeCost2(X, y, theta):
    m = len(y)
    sum = 0.0
    for i in range(0,m):
        hyp = theta[0,0] * X[i,0] + theta[1,0] * X[i,1] + theta[2,0] * X[i,2]
        sum += (hyp - y[i,0])**2
    res = (1/(2*m)) * sum
    return round(res,100)

# Computes cost function with theta (theta_0 to theta_n) as input value
# Returns float value
def computeCostN(X, y, theta):
    m = len(y)
    sum = 0.0
    for i in range(0,m):
        # The mathematical cost function assumes theta and xi are both column vectors
        # The data from the csv file makes and xi a row vector, so do theta instead of theta^T
        # hyp is a numpy 2d array
        hyp = np.dot(np.array([X[i,:]]), theta)
        sum += (hyp - y[i,0])**2
    res = (1/(2*m)) * sum[0,0]
    return round(res, 100)

# Finds the values of theta_0 and theta_1 that minimize the cost function
# theta and delta are vectors
def gradDesc1(X, y, theta, alpha, num_iters, returnJ):
    J = np.array([])
    theta = theta.astype(float)
    J = np.append(J, computeCostN(X, y, theta))
    m = len(y)
    for k in range(0,num_iters):
        delta = np.array([ [0],
                           [0] ])
        for i in range(0,m):
            hyp = np.dot(np.array([X[i, :]]), theta)
            delta[0,0] += (hyp - y[i,0]) * X[i,0]
            delta[1,0] += (hyp - y[i,0]) * X[i,1]
        theta -= alpha * (1/m) * delta
        J = np.append(J,computeCostN(X, y, theta))
    if returnJ == False:
        return theta
    else:
        return J

# Finds the values of theta_0 and theta_1 that minimize the cost function
# theta and delta are vectors
def gradDesc2(X, y, theta, alpha, num_iters, returnJ):
    J = np.array([])
    theta = theta.astype(float)
    J = np.append(J, computeCostN(X, y, theta))
    m = len(y)
    for k in range(0,num_iters):
        delta = np.array([ [0],
                           [0],
                           [0] ], dtype='float64')
        for i in range(0,m):
            hyp = np.dot(np.array([X[i, :]]), theta)
            delta[0,0] += (hyp - y[i,0]) * X[i,0]
            delta[1,0] += (hyp - y[i,0]) * X[i,1]
            delta[2,0] += (hyp - y[i,0]) * X[i,2]
        theta -= alpha * (1/m) * delta
        #print(computeCostN(X,y,theta))
        J = np.append(J,computeCostN(X, y, theta))
    if returnJ == False:
        return theta
    else:
        return J

def gradDescN(X, y, theta, alpha, num_iters, returnJ):
    theta = theta.astype(float)
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J = np.array([])
    J = np.append(J, computeCostN(X, y, theta))

    # Run this for a specified number of iterations
    for k in range(0, num_iters):
        # Initialize delta
        delta = np.zeros((theta.shape[0],1), dtype='float64')

        # Define delta
        for i in range(0, m):

            # Update delta_0 to delta_n+1
            for d in range(0,delta.shape[0]):
                delta[d,0] += ( h( np.array([X[i, :]]), theta ) - y[i, 0] ) * X[i, d]

        # Update theta
        theta -= alpha * (1 / m) * delta

        # Update cost using the theta corresponding to the current iteration
        J = np.append(J, computeCostN(X, y, theta))

    # Decide whether this function returns optimized theta or values of J
    if returnJ == False:
        return theta
    else:
        return J

# If there are too few training examples, function does not work well for some reason
# Only use if features differ by orders of magnitude
def featureNormalize(X):
    res = np.copy(X)
    for i in range(1,X.shape[1]):
        res[:, i] = ( res[:, i] - np.mean(res[:, i]) ) / np.std(res[:, i])
    return res

# X does not have to be feature normalized. However, when comparing the costs from using grad desc and normal
# eq, if grad desc uses normalized X, normal eq must use it too.
def normalEq(X,y):
    temp1 = lin.pinv( np.matmul(X.T,X) )
    temp2 = X.T
    temp3 = y
    return np.matmul( np.matmul(temp1,temp2), temp3 )
