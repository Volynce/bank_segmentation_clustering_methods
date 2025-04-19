from sklearn.cluster import KMeans

def kmeans_clustering(df_dr, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, init='k-means++', random_state=11)
    labels = kmeans.fit_predict(df_dr)
    
    return labels, kmeans
