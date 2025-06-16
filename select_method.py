import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans


def k_means_clustering(features, num_clusters):
    # features of size : NxF
    feature_type = 'numpy'
    if type(features) == torch.Tensor:
        feature_type = 'torch'
        dev = features.device
        features = features.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_

    if feature_type == 'torch':
        centroids = torch.from_numpy(centroids).to(dev)
    return centroids


def mini_batch_k_means(features, num_clusters, batch_size=1000, max_iterations=100):
    # features of size : NxF
    feature_type = 'numpy'
    if type(features) == torch.Tensor:
        feature_type = 'torch'
        dev = features.device
        features = features.detach().cpu().numpy()

    minibatch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=max_iterations, batch_size=batch_size)
    minibatch_kmeans.fit(features)
    centroids = minibatch_kmeans.cluster_centers_

    if feature_type == 'torch':
        centroids = torch.from_numpy(centroids).to(dev)
    return centroids


if __name__ == '__main__':
    # Example usage:
    feature_array = np.random.rand(100, 1024)  # 100 data points with 4 features each
    num_clusters = 10
    cluster_centers = k_means_clustering(feature_array, num_clusters)
    print(cluster_centers)

