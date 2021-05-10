from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(K, X):
    '''
    Uses the elbow method to find the optimal number of clusters for KMeans
    '''
    distortions = []
    for k in K:
        clusterer = KMeans(n_clusters=k)
        clusterer.fit(X)
        distortions.append(clusterer.inertia_)
        # print('Fon n_clusters = {} The inertia is : {} '.format(k, kmeans_model.inertia_))
    return distortions

def silhouette_method(K, X):
    '''
    Uses the silhouettes method to find the optimal number of clusters for KMeans
    '''
    silhouettes = []
    for k in K:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes.append(silhouette_avg)
        # print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

    return silhouettes