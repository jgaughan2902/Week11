from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import time
import pandas as pd

def kmeans(X, k):

    num_clusters = k

    km = KMeans(n_clusters = num_clusters, n_init = 'auto')
    km.fit(X)

    centroids = km.cluster_centers_

    labels = km.labels_

    return (centroids, labels)

# Globally accessible
diamonds = sns.load_dataset("diamonds")

numerical_cols = diamonds.select_dtypes(include = ['number']).columns

def kmeans_diamonds(n, k):
    
    subset = diamonds.head(n)

    X = subset[numerical_cols].values

    centroids, labels = kmeans(X, k)

    return (centroids, labels)

centroids, labels = kmeans_diamonds(n=1000, k=5)

def kmeans_timer(n, k, n_iter = 5):
    runtimes = []

    for i in range(n_iter):
        start_time = time.time()

        _ = kmeans_diamonds(n, k)

        end_time = time.time()

        duration = end_time - start_time

        runtimes.append(duration)

        return sum(runtimes) / len(runtimes)





