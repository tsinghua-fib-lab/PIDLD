import numpy as np


def generate_point_dataset(
    n_samples: int=1000,
    weights_true: np.ndarray=np.array([0.80, 0.20]),
    mu_true: np.ndarray=np.array([[5, 5], [-5, -5]]),
    cov_true: np.ndarray=np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
) -> np.ndarray:
    """
    Generate a point dataset from a Gaussian mixture model. Each point has a dimension of d.
    In the current implementation, d=2.

    Args:
        n_samples (int): The total number of samples (points) to generate.
        weights_true (np.ndarray): Ratio of number of points in each cluster. Shape: (n_components,).
        mu_true (np.ndarray): Means of the each cluster. Shape: (n_components, d).
        cov_true (np.ndarray): Covariances of the each cluster. Shape: (n_components, d, d).

    Returns:
        samples (np.ndarray): The generated data of shape (n_samples, d).
    """
    assert type(mu_true) == type(cov_true) == type(weights_true) == np.ndarray, "`mu_true`, `cov_true`, and `weights_true` must all be numpy arrays."
    assert mu_true.shape[0] == cov_true.shape[0] == weights_true.shape[0], "Number of components must match."
    assert mu_true.shape[1] == cov_true.shape[1], "Mean and covariance matrix must have the same dimension."
    assert cov_true.shape[1] == cov_true.shape[2], "Covariance matrix must be square."

    n_components = len(weights_true)

    samples = []
    for i in range(n_components):
        samples.append(np.random.multivariate_normal(mu_true[i], cov_true[i], size=int(n_samples*weights_true[i])))
    samples = np.concatenate(samples, axis=0) # Shape: (n_samples, d)
    np.random.shuffle(samples) # Shuffle the data
    return samples


if __name__ == '__main__': # For testing purposes only.
    import matplotlib.pyplot as plt
    
    n_samples = 1000
    mu_true = np.array([[5, 5], [-5, -5]]),
    cov_true = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
    weights_true = np.array([0.80, 0.20])

    data = generate_point_dataset(n_samples=n_samples, mu_true=mu_true, cov_true=cov_true, weights_true=weights_true) # Shape: (n_samples, 2)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=2)
    plt.show()
