from sklearn.cluster import SpectralClustering

def spectral_clustering(df_dr, n_clusters=3):
    sp_cluster = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    labels = sp_cluster.fit_predict(df_dr)
    
    return labels, sp_cluster