from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(df_dr, n_clusters=3):
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg_cluster.fit_predict(df_dr)
    
    return labels, agg_cluster
