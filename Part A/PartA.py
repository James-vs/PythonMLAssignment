import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.linear_model import LogisticRegression

# import numpy as np
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

trainingData = pandas.read_csv("TrainingDataBinary.csv", header=None)
testingData = pandas.read_csv("TestingDataBinary.csv", header=None)
# print(testingData)
print(testingData.shape)
print(trainingData.shape)
# X = trainingData.iloc[:, 0:117].drop([1,6,7,8,12,13,14,15,16,17,19,20,28,30,31,37,39,41,42,44,45,48,50,51,54,55,56,58,68,71,75,77,80,81,83,84,85,86,90,91,93,97,100,101,106,107,108,111,112,113,114],axis=1)
X = trainingData.iloc[:, 0:128]
X = preprocessing.scale(X)
y = trainingData.iloc[:, 128]
# print(X)


while True:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    print(model.score(X_test, y_test))
    print(model.predict(testingData))
