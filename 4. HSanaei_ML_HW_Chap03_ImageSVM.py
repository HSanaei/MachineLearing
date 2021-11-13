
from sklearn.datasets import fetch_lfw_people

"""Downloading LFW metadata: https://ndownloader.figshare.com/files/5976012
Downloading LFW metadata: https://ndownloader.figshare.com/files/5976009
Downloading LFW metadata: https://ndownloader.figshare.com/files/5976006
Downloading LFW data (~200MB): https://ndownloader.figshare.com/files/5976015"""

face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target
print('Input data size :', X.shape)

print('Output data size :', Y.shape)

print('Label names:', face_data.target_names)



for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')



import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
    xlabel=face_data.target_names[face_data.target[i]])

plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.svm import SVC
clf = SVC(class_weight='balanced', random_state=42)

parameters = {'C': [0.1, 1, 10], 'gamma': [1e-07, 1e-08, 1e-06], 'kernel' : ['rbf', 'linear'] }
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)

print('\n\n The best model:\n', grid_search.best_params_)


print('\n The best averaged performance:', grid_search.best_score_)


clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)

from sklearn.metrics import classification_report
print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%\n\n')
print(classification_report(Y_test, pred, target_names=face_data.target_names))





