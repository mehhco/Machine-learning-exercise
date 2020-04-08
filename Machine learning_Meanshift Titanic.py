import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
np.set_printoptions(threshold=np.inf)  # output all the data rather than shows...
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
import pandas as pd
# pd.set_option('display.max_columns', 1500)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = pd.read_excel("G:/edge下载/titanic.xls")
original_df = pd.DataFrame.copy(df) # cannot use original_df = df since if we do so, if df change, original_df will change either
df.drop(['body', 'name'], 1, inplace=True)  # 1 means drop the cotent of the column, if it is 0 means drops the label
df.apply(pd.to_numeric, errors='coerce') # change the data type to numeric type, if it is not applicapable, ignore it
df.fillna(0,inplace=True)


# print(df.head())
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]


n_clusters_ = len(np.unique(labels)) # how many unique labels (clusters) we have
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
print(original_df[original_df['cluster_group'] == 0].describe())










