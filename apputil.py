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
    # Define a subset containing only the n rows.
    subset = diamonds.head(n)

    # Define the array that goes in the kmeans function.
    X = subset[numerical_cols].values

    # Define the centroids and labels using the kmeans function.
    centroids, labels = kmeans(X, k)

    return (centroids, labels)

centroids, labels = kmeans_diamonds(n=1000, k=5)

def kmeans_timer(n, k, n_iter = 5):
    '''
    Runs the kmeans_diamonds() function and saves
    the runtime for each run.

    Parameters:
    n (int): The desired amount of rows for the 
    kmeans_diamonds() function.
    k (int): The number of clusters you want for the
    kmeans_diamonds() function
    n_iter (int): The amount of times you want the
    kmeans_diamonds() function to run.

    Return value:
    sum(runtimes) / len(runtimes): The average time
    across the n runs (in seconds).
    '''
    
    # Initiate a blank list for runtimes
    runtimes = []

    # Run a dummy call to improve consistency.
    if n_iter > 0:
            _ = kmeans_diamonds(10, 2)

    # Start a for loop for the amount of iterations.
    for _ in range(n_iter):
        
        # Store a beginning time for the iteration.
        start_time = time.time()

        # Run the k_means function (though we don't care
        # about the output in this case).
        _ = kmeans_diamonds(n, k)

        # Record the duration of the iteration.
        duration = time.time() - start_time

        # Append it to the runtimes list.
        runtimes.append(duration)

    return sum(runtimes) / len(runtimes)