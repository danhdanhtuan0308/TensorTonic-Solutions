import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    #Geting feature 
    w = np.zeros(X.shape[1])
    b = 0 
    # len of the data Number of samples 
    n_sample = len(X)
    #Loop 
    for epoch in range(steps) : 
        #Linear regression 
        z = np.dot(X,w)  + b  
        #Activation function map to -> 0 , 1 
        y_pred = _sigmoid(z)

        error =  y_pred - y

        #derivatives of weight and bias 
        dw = (1/n_sample) * np.dot(X.T,error)
        db = (1/n_sample) * sum(error)

        # update weight and bias 
        w = w - lr * dw
        b = b - lr * db
    
    return w, b 