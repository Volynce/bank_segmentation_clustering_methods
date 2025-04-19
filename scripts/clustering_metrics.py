from sklearn import metrics
import pandas as pd

def clustering_metrics(df_dr, df, algorithms):
    X, y = df_dr, df['TARGET (take a credit)']
    
    data = []
    for algo in algorithms:
        algo.fit(X)
        data.append(({
            'ARI': metrics.adjusted_rand_score(y, algo.labels_),
            'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
            'Homogenity': metrics.homogeneity_score(y, algo.labels_),
            'Completeness': metrics.completeness_score(y, algo.labels_),
            'V-measure': metrics.v_measure_score(y, algo.labels_),
            'Silhouette': metrics.silhouette_score(X, algo.labels_)}))
    
    results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                               'Completeness', 'V-measure',
                                               'Silhouette'],
                           index=['K-means', 'Affinity',
                                  'Spectral', 'Agglomerative'])
    
    return results
