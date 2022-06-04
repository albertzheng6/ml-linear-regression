from linReg import *

################################################## INPUT ##################################################

# Iterations
num_iters = 2000

# Learning rate
alpha = 0.01

# Read data
df = pd.read_csv('linReg1.csv')

################################################## DEFINE ##################################################

# Convert the appropriate data in dataframe to numpy 2d array
mat = df.loc[:,:].to_numpy()
mat = mat.astype(float)

'''
# Raw data does not have a predefined x_0 (which always equals one). The following can be used to create a new
# text file that adds an x_0 column to the data.
mat = np.insert(mat, 0, 1, axis=1)
np.savetxt("linReg1.csv", mat, delimiter=",")
'''

# X is a matrix of m training examples (each row) and n+1 features (each column)
X = getX(mat)
m = X.shape[0] # Number of training examples
n = X.shape[1] # Number of features (inc. x_0)

# Initial guess for theta (num of entries must equal num of features (inc. x_0) )
theta = np.zeros((X.shape[1],1))

# Feature normalize values if necessary (it usually is)
X_norm = featureNormalize(X)

# y is a column vector of all outputs y1 to ym, corresponding with each training example
y = getY(mat)

################################################## EXECUTE ##################################################

# Find theta such that error is minimized using gradient descent or the normal equation
# theta_min1 and theta_min2 should equal each other
theta_min1 = gradDescN(X, y, theta, alpha, num_iters, returnJ=False)
theta_min2 = normalEq(X,y)
print('Theta using gradient descent: ')
print(theta_min1)
print('Theta using normal equation: ')
print(theta_min2)

# Return cost for each iteration (should decrease over time)
J = gradDescN(X,y,theta,alpha,num_iters,returnJ=True)

# Make predictions
# x = np.array([ [x_0, x_1, ..., x_n] ])
x = np.array([ [1,3.5] ])
temp_X = np.append(X,x,axis=0)
temp_X = featureNormalize(temp_X)
x_norm = np.array([temp_X[-1]])
print('Predicted value using gradient descent: ' + str(h(x,theta_min1)) )
print('Predicted value using normal equation: ' + str(h(x,theta_min2)) )

########## PLOT ##########

# Plot linear regression (only if X has only 1 features, excluding x_0)
plt.scatter(X[:, 1], y, s=5)

x_reg = np.linspace(4, 25, 100)
y_reg1 = theta_min1[0] + theta_min1[1] * x_reg
y_reg2 = theta_min2[0] + theta_min2[1] * x_reg
plt.plot(x_reg, y_reg1, 'r', label='Gradient descent')
plt.plot(x_reg, y_reg2, 'b', label='Normal equation')

plt.style.use('seaborn-whitegrid')
plt.xlabel('xi')
plt.ylabel('y')
plt.title('xi vs y')
plt.xlim(4, 25)
plt.ylim(-5, 25)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Uncomment to plot speed of convergence
'''
x = np.arange(0,num_iters+1)
plt.plot(x,J, label='Cost')
#plt.plot(x,J, label='0.01')
#plt.plot(x,J, label='0.03')
#plt.plot(x,J, label='0.1')
#plt.plot(x,J, label='0.3')

plt.style.use('seaborn-whitegrid')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Speed of convergence')
plt.xlim(0,num_iters+1)
#plt.ylim(0, 25)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
'''
