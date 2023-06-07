from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

trainingData = pd.read_csv("TrainingDataMulti.csv", header=None)
testingData = pd.read_csv("TestingDataMulti.csv", header=None)
print(testingData.shape)
print(trainingData.shape)
X = trainingData.iloc[:, 0:128]
# X = preprocessing.scale(X)
y = trainingData.iloc[:, 128]
# print(trainingData[128].value_counts()) - proved that the data is balanced

# creating a RandomForestClassifier variable
rfc = RandomForestClassifier()
# using 30 as the random state value since this worked best in the SVC model

# creating a pipeline
# pipe = Pipeline(steps=[('scaler', MinMaxScaler()),('rfc', rfc)]) # StandardScaler()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
# 75% training and 25% test
# randomise/shuffle 30 data points when splitting the data


# defining a grid search and parameters for finding optimal hyperparameters
parameters = {
    'n_estimators': [500], # 50, 100, 200, 500, 1000
    'max_features': ['log2', 'sqrt'], # , 'sqrt'
    'criterion': ['entropy'], #, # 'gini', 'log_loss'
    'random_state': [30] # 20, 25, 35, 40, 42
} # 'rfc__' removed from beginning of variables since pipe no longer in use
cv_rfc = GridSearchCV(estimator=rfc, param_grid=parameters, n_jobs=-1, cv=10) # estimator=rfc removed for pipe testing
cv_rfc.fit(X_train, y_train) # .values.ravel()

# printing the accuracy and best parameters
print("Best parameter (CV score=%0.5f):" % cv_rfc.best_score_)
print(cv_rfc.best_params_)
print("Applying model to testing data")
y_pred = cv_rfc.predict(X_test)
print("Accuracy score: " + str(metrics.accuracy_score(y_test, y_pred)))
print("Testing Accuracy: " + str(cv_rfc.score(X_test, y_test)))
print("Applying Model to unseen data")
predLabels = cv_rfc.predict(testingData)
print(predLabels)

outputData = testingData
outputData[128] = predLabels
# outputData.to_csv("TestingResultsMulti.csv", header=False, index=False)

# while True:
# model = RandomForestClassifier()
# model.fit(X_train, y_train.values.ravel())

