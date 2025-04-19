import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from load_and_clean_data import load_and_clean_data
from elbow_method import elbow_method
from kmeans_clustering import kmeans_clustering
from agglomerative_clustering import agglomerative_clustering
from affinity_propagation import affinity_propagation
from spectral_clustering import spectral_clustering
from clustering_metrics import clustering_metrics

# Загрузка и очистка данных
df, df_dr = load_and_clean_data('../data/DATASET FOR CASE.csv')

# Метод локтя
elbow_method(df_dr)

# Кластеризация KMeans
kmeans_labels, kmeans_model = kmeans_clustering(df_dr)

# Агломеративная кластеризация
agg_labels, agg_model = agglomerative_clustering(df_dr)

# Аффинная кластеризация
aff_labels, aff_model = affinity_propagation(df_dr)

# Спектральная кластеризация
sp_labels, sp_model = spectral_clustering(df_dr)

# Оценка качества кластеризации
algorithms = [kmeans_model, aff_model, sp_model, agg_model]
results = clustering_metrics(df_dr, df, algorithms)

print(results)
