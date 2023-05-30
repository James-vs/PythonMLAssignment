from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

trainingData = pd.read_csv("TrainingDataBinary.csv", header=None)
testingData = pd.read_csv("TestingDataBinary.csv", header=None)
print(testingData.shape)
print(trainingData.shape)
X = trainingData.iloc[:, 0:128]
X = preprocessing.scale(X)
y = trainingData.iloc[:, 128]
# print(trainingData[128].value_counts()) - proved that the data is balanced

# creating an SVC classifier
clf = svm.SVC()

# creating a pipeline
pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svm', clf)])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
# 70% training and 30% test
# randomise/shuffle 30 data points when splitting the data


# defining a grid search and parameters for finding optimal hyperparameters
parameters = {
    "svm__kernel": ['rbf'], # 'linear', 'poly',
    "svm__C": [1, 1000, 1500, 2000], # 0.1, 0.01, 10,1, 100,  , 1e5 0.001,0.5e4, 1e4, 1.5e4
    "svm__gamma": [0.17], # 0.5, 0.1, , 0.025, 0.05
}
search = GridSearchCV(pipe, param_grid=parameters, n_jobs=-1, cv=10) # cv changed from 10
search.fit(X_train, y_train)

# printing the accuracy and best parameters
print("Best parameter (CV score=%0.5f):" % search.best_score_)
print(search.best_params_)

# while True:
# model = RandomForestClassifier()
# model.fit(X_train, y_train.values.ravel())
# print(model.score(X_test, y_test))
# print(model.predict(testingData))
