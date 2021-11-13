
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
wine_data = load_breast_cancer()
X = wine_data.data
Y = wine_data.target
print('Input data size :', X.shape)

print('Output data size :', Y.shape)

print('Label names:', wine_data.target_names)

n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()
print(f'{n_class0} class0 samples,\n{n_class1} class1 samples,\n{n_class2} class2 samples.')


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)

from sklearn.svm import SVC
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3,
gamma='auto_deprecated', kernel='linear', max_iter=-1,
probability=False, random_state=42, shrinking=True,
tol=0.001, verbose=False)

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

from sklearn.metrics import classification_report
pred = clf.predict(X_test)
print(classification_report(Y_test, pred))

