import numpy as np

"""
our linear regression
y_hat = b + w1x1 + w2x2 + ...+ w24x24
"""
class Model:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)#Start with one weight for each input feature
        self.bias = 0.0

    def predict(self, X):
        return X @ self.weights + self.bias #Predict lap times using the current weights


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

    for epoch in range(epochs):
        gradient_weights, gradient_bias = gradient(model, X, y_realized)
        model.weights -= learning_rate * gradient_weights
        model.bias -= learning_rate * gradient_bias
        y_predicted = model.predict(X)
        loss = mse(y_realized, y_predicted)
        losses.append(loss)
    return losses



def sgd(model, X, y_realized, learning_rate, epochs):
    losses = []
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        for i in rng.permutation(len(y_realized)):#iter thru all laps and update on each, need randomization to ensure no sequential bias of similar ordered data
            gradient_weights, gradient_bias = gradient(model, X[i:i+1], y_realized[i:i+1])# use features and time to calculate gradient, [i:i+1] selects one row keeps in array
            model.weights -= learning_rate * gradient_weights
            model.bias -= learning_rate * gradient_bias

        loss = mse(y_realized, model.predict(X))#take mse on all laps after epoch
        losses.append(loss)

    return losses