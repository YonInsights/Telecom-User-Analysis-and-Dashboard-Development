import numpy as np
import pandas as pd

def compute_experience_score(user_data, centroid):
    """
    Compute experience score based on Euclidean distance to the centroid of the worst experience cluster.

    Args:
        user_data (pd.DataFrame): User data for computing the score.
        centroid (np.array): Centroid of the worst experience cluster.

    Returns:
        pd.Series: Experience scores for each user.
    """
    # Compute the Euclidean distance between each user and the centroid
    distances = np.linalg.norm(user_data - centroid, axis=1)
    return distances
