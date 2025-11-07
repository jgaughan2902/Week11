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

# Exercise output that is required for this week

from apputil import *

%config InlineBackend.figure_formats = ['svg']
sns.set_theme(style="whitegrid")

n_values = np.arange(100, 50000, 1000)
k5_times = [kmeans_timer(n, 5, 20) for n in n_values]

k_values = np.arange(2, 50)
n10k_times = [kmeans_timer(10000, k, 10) for k in k_values]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.tight_layout()
fig.suptitle("KMeans Time Complexity", y=1.08, fontsize=14)

sns.lineplot(x=n_values, y=k5_times, ax=axes[0])
axes[0].set_xlabel("Number of Rows (n)")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_title('Increasing n for k=5 Clusters')

sns.lineplot(x=k_values, y=n10k_times, ax=axes[1])
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_title('Increasing k for n=10k Samples');