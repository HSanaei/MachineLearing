from sklearn.datasets import fetch_lfw_people

face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.decomposition import PCA

pca = PCA(n_components=100, whiten=True, random_state=42)

from sklearn.svm import SVC
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


model = Pipeline([('pca', pca), ('svc', svc)])
parameters_pipeline = {'svc__C': [1, 3, 10], 'svc__gamma': [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)

print('\n\n The best model:\n', grid_search.best_params_)


print('\n\n The best averaged performance:', grid_search.best_score_)

model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')


from sklearn.metrics import classification_report
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))


