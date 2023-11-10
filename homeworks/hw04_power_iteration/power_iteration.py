import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    eigenvalues, eigenvectors = np.linalg.eig(data)
    eigenval = float(np.max(np.abs(eigenvalues)))
    
    vec = np.ones(data.shape[1])
    for i in range(num_steps):
        vec = data.dot(vec)
        val = np.sqrt(np.sum(np.square(vec)))
        vec = vec/val
    
    return eigenval, vec