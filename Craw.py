from __future__ import print_function
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import itertools
import os
import os.path
import codecs

with open('articles.json') as json_data:
    articles = json.load(json_data)
    print(len(articles), "Articles loaded succesfully")

    typeLabel = []
    for article in articles:
        typeL = article["type"]
        typeLabel.append(typeL)
    typeLabel = set(typeLabel)
    typeLabel = list(set(typeLabel))

    #Make array Label is unique

    seen = set()
    resultType = []
    for item in typeLabel:
        if item not in seen:
            seen.add(item)
            resultType.append(item)

    x = 0
    countTrue = 0
    fileList = []
    Y = []

    for article in articles:
        for i in range(0, len(resultType)):
            if article["type"] == resultType[i]:
                filePathName = '{}_{}.txt'.format(i, x)
                
                with codecs.open(filePathName, 'w', encoding='utf-8') as reader:
                    json.dump(article["content"], reader)
                    linkPath = "/home/printf033c/PycharmProjects/knn/venv/" + filePathName
                    
                    if (os.path.exists(linkPath) == True):
                        fileList.append(linkPath)
                        Y.append(i)
                        countTrue += 1
                x = x + 1
        pass
    print(countTrue, "Article are converted completely")
    X = TfidfVectorizer(input="filename").fit_transform(fileList).toarray()


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=50)

    print("Training size: ", len(Y_train))
    print("Test size    : ", len(Y_test))

    n = 5
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    Y_predict = neigh.predict(X_test)

    print("Accuracy of 5NN: ", (100 * accuracy_score(Y_test, Y_predict)),"%")




