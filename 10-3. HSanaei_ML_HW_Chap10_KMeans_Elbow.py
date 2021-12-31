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
We perform k-means
clustering under different values of k on the iris data:
'''
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target


k_list = list(range(1, 7))
sse_list = [0] * len(k_list)


'''
We use the whole feature space and k ranges from 1 to 6. Then, we train individual
models and record the resulting SSE, respectively:
'''
for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)

        sse += np.linalg.norm(X[cluster_i] - centroids[i])

    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse

'''
Finally, we plot the SSE versus the various k ranges, as follows:
'''
plt.plot(k_list, sse_list)
plt.show()
