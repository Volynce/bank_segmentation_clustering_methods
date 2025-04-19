import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def elbow_method(df_dr):
    inertia = []
    k_range = range(1, 15)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_dr)
        inertia.append(kmeans.inertia_)
    
    # Построение графика метода локтя
    plt.figure(figsize=(15, 10))
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.title('Метод локтя для определения количества кластеров')
    plt.show()
