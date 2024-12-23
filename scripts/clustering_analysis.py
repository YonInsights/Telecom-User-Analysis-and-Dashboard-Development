from sklearn.cluster import KMeans

def perform_clustering(data, n_clusters=2):
    """
    Perform K-means clustering.

    Args:
        data (pd.DataFrame): Data for clustering.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: Fitted KMeans model, cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return kmeans, labels
