'''
Python Machine Learning
Teacher: Dr Rahmani
Student: Hossein SANAEI ~حسین سنایی
Homework Chapter 10

Aras International Campus of University of Tehran
Fall 1400 (2021)
GitHub: https://github.com/HSanaei/MachineLearing.git

Chapter 10  Discovering Underlying Topics in the Newsgroups Dataset
with Clustering and Topic Modeling

'''


'''
We will use the iris dataset from scikit-learn as an example.
Let's first load the data and visualize it. We herein only use two features
out of the original four for simplicity:
'''
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target


'''
Since the dataset contains three iris classes, we plot it in three different colors,
as follows:
'''
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


'''
Assuming we know nothing about the label y, we try to cluster the data into three
groups, as there seem to be three clusters in the preceding plot (or you might say
two, which we will come back to later). Let's perform step 1, specifying k, and step 2,
initializing centroids, by randomly selecting three samples as initial centroids:
'''
k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]

'''
We visualize the data (without labels any more) along with the initial random
centroids:
'''
def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()

visualize_centroids(X, centroids)

'''
Now we perform step 3, which entails assigning clusters based on the nearest
centroids. First, we need to define a function calculating distance that is measured
by the Euclidean distance, as demonstrated herein:
'''
def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

'''
Then, we develop a function that assigns a sample to the cluster of the nearest
centroid:
'''
def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster

'''
With the clusters assigned, we perform step 4, which involves updating the centroids
to the mean of all samples in the individual clusters:
'''
def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)

clusters = np.zeros(len(X))

'''
Finally, we have step 5, which involves repeating step 3 and step 4 until the model
converges and whichever of the following occurs:
• Centroids move less than the pre-specified threshold
• Sufficient iterations have been taken
We set the tolerance of the first condition and the maximum number of iterations
as follows:
'''
tol = 0.0001
max_iter = 100


'''
Initialize the clusters' starting values, along with the starting clusters for all samples
as follows:
'''
iter = 0
centroids_diff = 100000


'''
With all the components ready, we can train the model iteration by iteration where
it first checks convergence, before performing steps 3 and 4, and then visualizes the
latest centroids:
'''
from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids:\n', centroids)
    print('Centroids move: {:5.4f}'.format(centroids_diff))
    visualize_centroids(X, centroids)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()
