from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

trainingData = pd.read_csv("TrainingDataBinary.csv", header=None)
testingData = pd.read_csv("TestingDataBinary.csv", header=None)
print(testingData.shape)
print(trainingData.shape)
X = trainingData.iloc[:, 0:128]
X = preprocessing.scale(X)
y = trainingData.iloc[:, 128]
# print(trainingData[128].value_counts()) - proved that the data is balanced

# creating a RandomForestClassifier variable
rfc = RandomForestClassifier(random_state=30)
# using 30 as the random state value since this worked best in the SVC model

# creating a pipeline
# pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svm', clf)])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
# 70% training and 30% test
# randomise/shuffle 30 data points when splitting the data


# defining a grid search and parameters for finding optimal hyperparameters
parameters = {
    'n_estimators': [100, 200, 500], #
    'max_features': ['sqrt', 'log2']# ,
    ##'max_depth': [4, 5, 6, 7, 8],
    ##'criterion': ['gini', 'entropy']
}
cv_rfc = GridSearchCV(estimator=rfc, param_grid=parameters, n_jobs=-1, cv=10) # 'pipe,' removed for rfc testing
cv_rfc.fit(X_train, y_train)

# printing the accuracy and best parameters
print("Best parameter (CV score=%0.5f):" % cv_rfc.best_score_)
print(cv_rfc.best_params_)

# while True:
# model = RandomForestClassifier()
# model.fit(X_train, y_train.values.ravel())
# print(model.score(X_test, y_test))
# print(model.predict(testingData))
