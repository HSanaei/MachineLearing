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
First, import the KMeans class and initialize a model with three clusters, as
follows:
'''

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

from matplotlib import pyplot as plt

'''
Assuming we know nothing about the label y, we try to cluster the data into three
groups, as there seem to be three clusters in the preceding plot (or you might say
two, which we will come back to later). Let's perform step 1, specifying k, and step 2,
initializing centroids, by randomly selecting three samples as initial centroids:
'''
k = 3
from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_


plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()
