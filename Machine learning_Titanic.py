import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np

np.set_printoptions(threshold=np.inf)  # output all the data rather than shows...
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

pd.set_option('display.max_columns', 1000)
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()

df = pd.read_excel("G:/edge下载/titanic.xls")
df.drop(['body', 'name'], 1, inplace=True)  # 1 means drop the cotent of the column, if it is 0 means drops the label
df.apply(pd.to_numeric, errors='coerce') # change the data type to numeric type, if it is not applicapable, ignore it
df.fillna(0,inplace=True)


print(df.head())
def handle_non_numerical_values(df):  # handle the rest non numerical value in the data frame
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_cotents = df[column].values.tolist()  # store the non-number content to a list
            unique_content = set(column_cotents)  # store the content as a set
            x = 0
            for content in unique_content:  # assign every unique content a unique number
                if content not in text_digit_vals:
                    text_digit_vals[content] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df


df = handle_non_numerical_values(df)
df.drop(['sex'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct+=1

print(correct/len(X))