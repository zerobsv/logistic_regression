import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generateData(cols=50):
    """
    Generates data where y is 0 for x < 3 and 1 for x >= 3.
    :param cols: Number of data points to generate.
    :return: X, y, and initial theta.
    """
    # X values from 0 to 10
    x_values = np.linspace(0, 10, cols)
    y = np.zeros(shape=(cols, 1))
    
    # Create y as a binary vector based on the condition x >= 3
    y = np.array((x_values >= 3).astype(int))
    
    # The shape will be (m, 2) where m is cols.
    X = np.stack([np.ones(cols), x_values], axis=1)

    # Initialize theta with zeros for 2 parameters (bias and feature)
    theta = np.zeros(shape=(2, 1))
    
    print("X shape:", X.shape, "y shape:", y.shape, "theta shape:", theta.shape)
    return X, y, theta


if __name__ == "__main__":
    # Corrected: Call generateData with a reasonable number of points
    X, y, theta = generateData(cols=100)

    alpha = 0.05
    numIter = 1000
    lambda_ = 0.05

    log_reg = LogisticRegression(C=lambda_, max_iter=numIter)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    log_reg.fit(X_train, y_train)

    theta_final = log_reg.coef_.reshape(-1, 1)

    print("theta_final: ", theta_final, theta_final.shape)

    y_pred_proba = log_reg.predict_proba(X_test)
    y_pred = np.array((y_pred_proba >= 0.5).astype(int))

    print("y_pred_proba: ", y_pred_proba)
    print("y_pred: ", y_pred)

    X_value = X_test[:, 1]

    for x, pred, actual in zip(X_value, y_pred, y_test):
        print(f"XValue: {x}, Predicted: {pred[1] if pred[1]==1 else 0}, Actual: {actual}")
