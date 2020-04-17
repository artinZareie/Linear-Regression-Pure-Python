"""
Linear Regression Without Scikit Learn Regressor
"""


# Importing Used Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionModel:
    
    # Constructor
    # Takes three parameters, First parameter (alpha) is learning rate and second one (lbd) is lambda
    # And thirth one (delta) is function to control ending GD. when derivative is smaller than delta
    # GD will stop.
    def __init__(self, alpha=0.01, lbd=0.01, delta=0.001):
        self._alpha = alpha
        self._lambda = lbd
        self._delta = delta
        
    # Fits moedl with data    
    def fit(self, X, y):
        self.m = y.shape[0] # Count of records
        self.n = X.shape[1] # Count of features
        self.X = X
        self.y = y
        self.theta = np.zeros((self.n, 1))
        self.gd() # Run gradient descent
        return self.theta
        
    # Predicts X by (X * theta)
    def predict(self, X):
        return np.dot(X, self.theta)
    
    # Returns Mean Squear Error
    def computeCost(self):
        predictions = self.predict(self.X) # Predicted values
        return ((1 / (2 * self.m)) * np.sum(np.power(predictions - self.y, 2))) + () # Calcute 2nd power of (h - y) and gets 1/2 of its mean.
   
    # partial derivative of cost function
    # Gets one parameter. that is wich part do you wnat to derivative. for example if you give 1 it means
    # Partial derivative by theta1
    def derivative(self, i):
        predictions = self.predict(self.X) # Predicted values
        cost = ((1 / self.m) * np.sum((predictions - self.y) * self.X[:, i]))
        if not i == 0:
            cost += (self._lambda * self.theta[i, :])
        return cost
    
    # Gradient Descent
    def gd(self):
        cond = False # Condition to break while
        for i in range(0, self.n):
            if abs(self.derivative(i)) > self._delta:
                cond = True
        while cond:
            sim_theta = self.theta # Simulated theta to update simulately
            for i in range(0, self.n):
                sim_theta[i, :] -= self._alpha * self.derivative(i) # Theta minus gradient
            self.theta = sim_theta
            cond = False
            for i in range(0, self.n):
                if abs(self.derivative(i)) > self._delta:
                    cond = True

# Defining Constatnts (Learning Rate and Lambda)
Alpha = 0.0001 # Learning Rate
Lambda = 0.01

# Load data
dataset = pd.read_csv("train.csv")
X = np.array(dataset.drop("y", 1))
y = np.array([dataset["y"]]).T

# Create an instance of LinearRegressionModel
model = LinearRegressionModel(alpha=Alpha, lbd=Lambda, delta=0.0000000001)
model.fit(X, y)

plt.scatter(X, y)
plt.plot(X, model.predict(X))