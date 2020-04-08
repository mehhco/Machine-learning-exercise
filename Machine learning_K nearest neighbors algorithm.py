from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        Warning.warn('k is set to a value less than total voting groups')

    distance = []
    for group in data:
        for feature in data[group]:
            # euclidean_distance = np.sqrt( np.sum((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2  ))
            euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distance.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence


Accuracy=[]
for i in range(5):
    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)  #洗牌

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print('Accuracy', correct/total)
    Accuracy.append(correct/total)
print(sum(Accuracy)/len(Accuracy))