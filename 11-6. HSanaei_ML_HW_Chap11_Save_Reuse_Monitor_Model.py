'''
Python Machine Learning
Teacher: Dr Rahmani
Student: Hossein SANAEI ~حسین سنایی
Homework Chapter 10

Aras International Campus of University of Tehran
Fall 1400 (2021)
GitHub: https://github.com/HSanaei/MachineLearing.git

Chapter 11  Word Save Reuse Monitor Model

'''

from sklearn import datasets
dataset = datasets.load_diabetes()
X, y = dataset.data, dataset.target

num_new = 30    # the last 30 samples as new data set
X_train = X[:-num_new, :]
y_train = y[:-num_new]
X_new = X[-num_new:, :]
y_new = y[-num_new:]


# Data pre-processing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

import pickle
# Save the scaler
pickle.dump(scaler, open("scaler.p", "wb" ))

X_scaled_train = scaler.transform(X_train)


# Regression model training
from sklearn.svm import SVR
regressor = SVR(C=20)
regressor.fit(X_scaled_train, y_train)
# Save the regressor
pickle.dump(regressor, open("regressor.p", "wb"))


# Deployment
my_scaler = pickle.load(open("scaler.p", "rb" ))
my_regressor = pickle.load(open("regressor.p", "rb"))

X_scaled_new = my_scaler.transform(X_new)
predictions = my_regressor.predict(X_scaled_new)


# Monitor
from sklearn.metrics import r2_score
print(f'Health check on the model, R^2: {r2_score(y_new, predictions):.3f}')
