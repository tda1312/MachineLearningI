import numpy as np

x = np.array([[30, 3, 6], [43, 4, 8], [25, 2, 3], [51, 4, 9], [40, 3, 5], [20, 1, 2]])
x_w1 = np.c_[np.ones((len(x), 1)), x]
y = np.array([2.5, 3.4, 1.8, 4.5, 3.2, 1.6])
y.shape += (1,)

def hypothesis(x, y, theta):
    return np.dot(x, theta)

def gradient(x, y, theta, m):
    h = hypothesis(x, y, theta)
    grad = np.dot(x.T, (h - y))
    
    return grad

def cost_cal(x, y, theta):
    h = hypothesis(x, y, theta)
    cost = np.sum(np.square(h - y))
    
    return cost

def stochastic_learning(x, y, theta, m):
    epochs = 50
    rate = 0.00001
    
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            x_i = x[rand_index:rand_index+1]
            y_i = y[rand_index:rand_index+1]
            
            theta = theta - (rate * gradient(x_i, y_i, theta, m))
            cost = cost_cal(x_i, y_i, theta)
            
    print("\n------STOCHASTIC LEARNING------")
    print('Theta:\n\n{}'.format(theta))
    print('\nCost:\n\n{}'.format(cost))
            
def batch_learning(x, y, theta, n, m):
    epochs = 100
    rate = 0.00001
    
    print("\n------BATCH LEARNING------")
    for i in range(epochs):
        theta = theta - (rate * (1 / n) * gradient(x, y, theta, m))
        cost = (1 / (2 * n)) * cost_cal(x, y, theta)
        
    print("Theta:\n\n{}".format(theta))
    print('\nCost:\n\n{}'.format(cost))

def minibatch_learning(x, y, theta, m):
    epochs = 100
    minibatch_size = 2
    rate = 0.00001
    
    for epoch in range(epochs):
        shuffled_features = np.random.permutation(m)
        x_shuffled = x[shuffled_features]
        y_shuffled = y[shuffled_features]
        
        for i in range(0, m, minibatch_size):
            x_i = x_shuffled[i:i+minibatch_size]
            y_i = y_shuffled[i:i+minibatch_size]
            
            theta = theta - (rate * (2 / minibatch_size) * gradient(x_i, y_i, theta, m))
            cost = (1 / minibatch_size) * cost_cal(x_i, y_i, theta)
    
    print("\n------MINIBATCH LEARNING------")
    print("Theta:\n\n{}".format(theta))
    print("\nCost:\n\n{}".format(cost))

def normal_equation(x, y, theta):
    theta = np.linalg.inv(x.T.dot(x))
    theta = theta.dot(x.T)
    theta = theta.dot(y)
    
    print("\n------Normal equation------")
    print("Theta:\n\n{}".format(theta))
    
def gradient_descent(x, y):
    
    m = x.shape[1]
    theta = np.zeros((m, 1))
    n = len(x_w1)
    
    batch_learning(x, y, theta, n, m)
    stochastic_learning(x, y, theta, m)
    minibatch_learning(x, y, theta, m)
    normal_equation(x, y, theta)
    
gradient_descent(x_w1, y)