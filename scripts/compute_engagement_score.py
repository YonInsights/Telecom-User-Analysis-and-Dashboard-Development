import numpy as np
import pandas as pd

def compute_engagement_score(user_data, centroid):
    """
    Compute engagement score based on Euclidean distance to the centroid of the less engaged cluster.

    Args:
        user_data (pd.DataFrame): User data for computing the score.
        centroid (np.array): Centroid of the less engaged cluster.

    Returns:
        pd.Series: Engagement scores for each user.
    """
    # Compute the Euclidean distance between each user and the centroid
    distances = np.linalg.norm(user_data - centroid, axis=1)
    return distances
