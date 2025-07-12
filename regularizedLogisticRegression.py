import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid function.
    :param z: Input value or array.
    :return: Sigmoid of z.
    """
    # The sigmoid function correctly implemented
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y, lambda_):
    """
    Compute the cost and gradient for regularized logistic regression.
    :param theta: Parameters for the logistic regression model.
    :param X: Input data.
    :param y: Labels for the input data.
    :param lambda_: Regularization parameter.
    :return: Cost value.
    """
    m = X.shape[0]
    # X has shape (m, n) and theta has shape (n, 1), so the result is (m, 1).
    htheta = sigmoid(np.dot(X, theta))

    # Cost function with correct regularization (excluding theta[0])
    cost = (1 / m) * np.sum(-y * np.log(htheta) - (1 - y) * np.log(1 - htheta))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2) # Regularize all but the bias term
    
    return cost + reg_term

def gradient_descent(theta, X, y, lambda_, alpha, numIter):
    """
    Perform gradient descent to find the optimal theta.
    :param theta: Parameters for the logistic regression model.
    :param X: Input data.
    :param y: Labels for the input data.
    :param lambda_: Regularization parameter.
    :param alpha: Learning rate.
    :param numIter: Number of iterations.
    :return: The final optimized theta.
    """
    m = X.shape[0]
    for _ in range(numIter):
        htheta = sigmoid(np.dot(X, theta))
        
        # Calculate the gradient term
        gradient_non_reg = (1 / m) * np.dot(X.T, (htheta - y))
        
        # Calculate the regularization term for the gradient
        reg_term_grad = (lambda_ / m) * np.copy(theta)
        reg_term_grad[0] = 0 # Do not regularize the bias term's gradient
        
        gradient = gradient_non_reg + reg_term_grad
        
        theta = theta - alpha * gradient
    
    return theta

def generateData(cols=50):
    """
    Generates data where y is 0 for x < 3 and 1 for x >= 3.
    :param cols: Number of data points to generate.
    :return: X, y, and initial theta.
    """
    # X values from 0 to 10
    x_values = np.linspace(0, 10, cols)
    
    # Create y as a binary vector based on the condition x >= 3
    y_values = (x_values >= 3).astype(int)
    
    # The shape will be (m, 2) where m is cols.
    X = np.stack([np.ones(cols), x_values], axis=1)
    
    # Reshape y to a column vector
    y = y_values.reshape(-1, 1)
    
    # Initialize theta with zeros for 2 parameters (bias and feature)
    theta = np.zeros(shape=(2, 1))
    
    print("X shape:", X.shape, "y shape:", y.shape, "theta shape:", theta.shape)
    return X, y, theta


"""
Idea is that the data contains a single feature whose value is 1 after
the point x=3, and 0 before that.

The goal is to predict whether the value of the feature is 1 or 0 given
the x value. And a logistic regression model can be used to do this.

"""

if __name__ == "__main__":
    # Corrected: Call generateData with a reasonable number of points
    X, y, theta = generateData(cols=100)

    alpha = 0.05
    numIter = 1000
    lambda_ = 0.05

    beforeLearn = cost_function(theta, X, y, lambda_)
    print("Cost before learning: ", beforeLearn)
    
    theta_final = gradient_descent(theta, X, y, lambda_, alpha, numIter)
    
    afterLearn = cost_function(theta_final, X, y, lambda_)
    print("Cost after learning: ", afterLearn)
    
    y_pred_proba = sigmoid(np.dot(X, theta_final))
    
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("Predicted theta values:", theta_final)
    print("Sample predictions vs. actual values:")
    
    # Check the predictions for a few points
    for i in [10, 20, 30, 40, 50, 60, 70, 80]:
        print(f"X value: {X[i, 1]:.2f}, Predicted: {y_pred[i, 0]}, Actual: {y[i, 0]}")
