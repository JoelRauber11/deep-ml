import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape                  # m = Anzahl Punkte, n = Anzahl Gewichte
    y = y.reshape(-1, 1)            # y als Splate, damit die Rechnungen passen
    theta = np.zeros((n, 1))        # Start = alle Gewichte 0

    for _ in range(iterations):
        pred = X @ theta            # 1. Was sagt die Gerade voraus?
        errors = pred - y           # 2. Wie weit daneben?
        grad = (X.T @ errors) / m  # 3. In welche Richtung korrigieren?
        theta -= alpha * grad       # 4. Kleiner Schritt in die richtige Richtung

    return np.round(theta.flatten(), 4)