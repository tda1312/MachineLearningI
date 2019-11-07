import numpy as np

x = np.array([[30, 3, 6], [43, 4, 8], [25, 2, 3], [51, 4, 9], [40, 3, 5], [20, 1, 2]])
x_w1 = np.c_[np.ones((len(x), 1)), x]
y = np.array([2.5, 3.4, 1.8, 4.5, 3.2, 1.6])
y.shape += (1,)

def hypothesis(theta):
    return np.dot(x_w1, theta)

def gradient(x, y, theta, m):
    h = hypothesis(theta)
    grad = np.dot(x.T, (h - y))
    
    return grad

def cost_cal(x, y, theta):
    h = hypothesis(theta)
    cost = np.sum(np.square(h - y))
    
    return cost

def stochastic_learning(x, y, theta, m):
    epochs = 50
    rate = 0.001
    
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            x_i = x[rand_index:rand_index+1]
            y_i = y[rand_index:rand_index+1]
            
            theta = theta - (rate * gradient(x_i, y_i, theta, m))
            cost = cost_cal(x_i, y_i, theta)
            
    print("------STOCHASTIC LEARNING------")
    print('Theta:\n{}'.format(theta))
    print('Cost: {}'.format(cost))
            
def batch_learning(x, y, theta, n, m):
    epochs = 100
    rate = 0.00001
    
    print("------BATCH LEARNING------")
    for i in range(epochs):
        theta = theta - (rate * (1 / n) * gradient(x, y, theta, m))
        cost = (1 / (2 * n)) * cost_cal(x, y, theta)
        
    print('Theta:')
    print("\n{}".format(theta))
    print('Cost: {}'.format(cost))

# mini_batch_learning():
    

def gradient_descent(x, y):
    
    m = x.shape[1]
    theta = np.zeros((m, 1))
    n = len(x_w1)
    
    batch_learning(x, y, theta, n, m)
    stochastic_learning(x, y, theta, m)
    
gradient_descent(x_w1, y)