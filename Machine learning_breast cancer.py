import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import scrapy

df = pd.read_csv('G:/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))  # not including the class column
y = np.array(df['class'])  # only the class column

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf = svm.SVC()
clf.fit(X_train, y_train)

accurary = clf.score(X_test, y_test)
print(accurary)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 5, 1, 2, 3, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
