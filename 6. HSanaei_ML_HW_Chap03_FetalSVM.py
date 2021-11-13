
from sklearn.datasets import fetch_lfw_people
face_data = fetch_lfw_people(min_faces_per_person=80)

import pandas as pd
df = pd.read_excel('CTG.xls', "Raw Data")

X = df.iloc[1:2126, 3:-2].values
Y = df.iloc[1:2126, -1].values

print(Counter(Y))

from sklearn.svm import SVC
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


svc = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5), 'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)



svc_best = grid_search.best_estimator_
accuracy = svc_best.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')


prediction = svc_best.predict(X_test)
report = classification_report(Y_test, prediction)
print(report)