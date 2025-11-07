from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import time
import pandas as pd

def kmeans(X, k):
    '''
    Performs k-means clustering on X.

    Parameters:
    X (np.array): A numerical array.
    k (int): The number of clusters you want.

    Return value:
    (centroids, labels) (tuple): A tuple containing
    a 2D array of the cluster centroids and a 1D array
    of the index of the assigned cluster for each row in X.
    '''
    # Create a variable from the desired cluster input.
    num_clusters = k

    # Create an object using the KMeans built-in function.
    km = KMeans(n_clusters = num_clusters, n_init = 'auto')

    # Fit using the numerical array input.
    km.fit(X)

    # Define the centroids using a built in method.
    centroids = km.cluster_centers_

    # Define the labels using a built in method.
    labels = km.labels_

    return (centroids, labels)

# Load the diamonds data set and make it globally accessible.
diamonds = sns.load_dataset("diamonds")

# Find only the numerical columns in the diamonds data set
# and make it globally available.
numerical_cols = diamonds.select_dtypes(include = ['number']).columns

def kmeans_diamonds(n, k):
    '''
    Runs the kmeans function to create k clusters
    on the first n rows of the numeric diamonds dataset.

    Parameters:
    n (int): The desired amount of rows.
    k (int): The number of clusters you want.

    Return value:
    (centroids, labels) (tuple): A tuple containing
    a 2D array of the cluster centroids and a 1D array
    of the index of the assigned cluster for each row in X.
    '''
    # Define a subset containing only the 
    #subset = diamonds.head(n)
    subset = numerical_cols.head(n)

    X = subset.values

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
