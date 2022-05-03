import numpy as np


class LinearRegression:
    """
    Linear Regression Using Gradient Descent.
    Parameters
    ----------
    lr : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    weights : weights/ after fitting the model
    losses : total error of the model after each iteration
    """

    def __init__(self, lr=0.005, n_iterations=1000, limit=10):
        # Store the learning rate and the number of iterations
        # use self.lr and self.n_iterations
        print("Initializing Linear Regression")
        
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = np.zeros((2, 1))
        self.losses = []
        self.limit = limit

    def fit(self, X, y):
        """
        Fit the training data
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        """

        # Intilize the weights and losses

        # Implement the gradient descent algorithm
        
        for _ in range(self.n_iterations):
            # Predict the output
            y_pred = np.dot(X, self.weights)
            # AKA X[0]weights[0] + X[1] * weights[1]
            
            # Calculate the residuals (residuals = y_pred - y)
            residuals = y_pred - y #pontok egyenestol valo tavolsaga

            # Calculate the gradient
            gradient_vector = np.dot(X.T, residuals)
            #mennyit kene valtoztatni az egyenesen hogy javuljon

            # Store loss (summ of residuals squared)
            loss = np.sum((residuals ** 2))
            self.losses.append(loss)

            # Update the weights
            self.weights -=self.lr*gradient_vector
            
            if loss<self.limit:
                print('early stop')
                break
            
        print('Model has been trained')    

    def predict(self, X):
        """Predicts the value after the model has been trained.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        prediction = None
        y_pred = np.dot(X, self.weights)
        
        # Predict the output

        return prediction
