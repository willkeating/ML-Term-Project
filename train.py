import numpy as np
import time

"""
our linear regression
y_hat = b + w1x1 + w2x2 + ...+ w24x24
"""
class Model:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def predict(self, X):
        return X @ self.weights + self.bias


""""
    calc the MSE 
    = (1/n) * sum((y_predicted - y_realized)^2)
"""
def mse(y_realized, y_predicted):
    error = y_predicted - y_realized
    squared_error = error**2
    return np.mean(squared_error)# 1/n sum piece


"""
calculate gradient of mse - helper for GD and SGD

calc gradient for each weight and another for the bias
    
"""
def gradient(model, X, y_realized):
    y_predicted = model.predict(X)
    n = len(y_realized)
    gradient_weights = (2/n)* (X.T @ (y_predicted - y_realized)) # dL/dw=(2/n) * X.T @ (y_predicted - y_realized)
    gradient_bias = (2/n) * np.sum(y_predicted - y_realized) #dL/db=(2/n)* sum(y_predicted - y_realized)
    return gradient_weights, gradient_bias





"""
    calc gradient DESCENT
    
    repeatedly adjust weights and bias to min mse
"""
def gd(model, X, y_realized, learning_rate, epochs):
    losses = []
    times = []
    start = time.perf_counter()

    for epoch in range(epochs):
        gradient_weights, gradient_bias = gradient(model, X, y_realized)
        model.weights -= learning_rate * gradient_weights
        model.bias -= learning_rate * gradient_bias

        y_predicted = model.predict(X)
        loss = mse(y_realized, y_predicted)
        losses.append(loss)
        times.append(time.perf_counter() - start)
    return losses, times


"""
SGD function
    
        

"""
def sgd(model, X, y_realized, learning_rate, epochs):
    losses = []
    times = []
    start = time.perf_counter()
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        for i in rng.permutation(len(y_realized)):#iter thru all laps and update on each, need randomization to ensure no sequential bias of similar ordered data
            gradient_weights, gradient_bias = gradient(model, X[i:i+1], y_realized[i:i+1])# use features and time to calculate gradient, [i:i+1] selects one row keeps in array
            model.weights -= learning_rate * gradient_weights
            model.bias -= learning_rate * gradient_bias

        loss = mse(y_realized, model.predict(X))#take mse on all laps after epoch
        losses.append(loss)
        times.append(time.perf_counter() - start) #time counter


    return losses, times


"""
    Calculate momentum
    
    v_t = B_v_t-1 + grad(MSE)

"""
def momentum(model, X, y_realized, learning_rate, epochs, beta = 0.9):
    losses = []#mse per epoch

    #initialize momentum for weights & bias
    velocity_weights = np.zeros_like(model.weights)
    velocity_bias = 0.0

    for epoch in range(epochs):
        gradient_weights, gradient_bias = gradient(model, X, y_realized)
        velocity_weights = beta * velocity_weights + gradient_weights
        velocity_bias = beta * velocity_bias + gradient_bias
        model.weights -= learning_rate * velocity_weights
        model.bias -= learning_rate * velocity_bias

        y_predicted = model.predict(X)
        loss = mse(y_realized, y_predicted)
        losses.append(loss)

    return losses


def adam(model, X, y_realized, learning_rate, epochs, beta1 = 0.9, beta2 =0.999, epsilon=1e-8):
    losses = []

    m_weights = np.zeros_like(model.weights)#first moment/moving avg of grads
    m_bias = 0.0

    v_weights = np.zeros_like(model.weights)#second moment
    v_bias = 0.0

    for epoch in range(epochs):
        gradient_weights, gradient_bias = gradient(model, X, y_realized)

        m_weights = beta1 * m_weights + (1-beta1)* gradient_weights**2 # first moment
        m_bias = beta1 * m_bias + (1-beta1) * gradient_bias

        v_weights = beta2 * v_weights + (1- beta2)*gradient_weights**2# 2nd moment
        v_bias = beta2 * v_bias +  (1-beta2) * gradient_bias**2

        t = epoch + 1 #correct bias from starting at zero

        m_weights_hat = m_weights / (1-beta1**t)
        m_bias_hat = m_bias / (1 - beta1**t)

        v_weights_hat = v_weights / (1-beta2**t)
        v_bias_hat = v_bias / (1-beta2**t)

        model.weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
        model.bias -= learning_rate * m_bias_hat / (np.sqrt(v_bias_hat) + epsilon)

        # Calculate MSE after each epoch
        y_predicted = model.predict(X)
        loss = mse(y_realized, y_predicted)
        losses.append(loss)

    return losses
