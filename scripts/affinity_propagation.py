from sklearn.cluster import AffinityPropagation

def affinity_propagation(df_dr):
    aff_cluster = AffinityPropagation()
    labels = aff_cluster.fit_predict(df_dr)
    
    return labels, aff_cluster
