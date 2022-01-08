'''
Python Machine Learning
Teacher: Dr Rahmani
Student: Hossein SANAEI ~حسین سنایی
Homework Chapter 10

Aras International Campus of University of Tehran
Fall 1400 (2021)
GitHub: https://github.com/HSanaei/MachineLearing.git

Chapter 11  Pricipal Component Analysis (PCA)

'''

from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
# Keep different number of top components
N = [10, 15, 25, 35, 45]
for n in N:
    pca = PCA(n_components=n)
    X_n_kept = pca.fit_transform(X)
    # Estimate accuracy on the data set with top n components
    classifier = SVC(gamma=0.005)
    score_n_components = cross_val_score(classifier, X_n_kept, y).mean()
    print(f'Score with the dataset of top {n} components: {score_n_components:.2f}')