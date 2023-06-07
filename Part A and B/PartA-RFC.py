from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# importing the training and testing data into DataFrames
trainingData = pd.read_csv("TrainingDataBinary.csv", header=None)
testingData = pd.read_csv("TestingDataBinary.csv", header=None)
print(testingData.shape)
print(trainingData.shape)
X = trainingData.iloc[:, 0:128]
y = trainingData.iloc[:, 128]

# creating a RandomForestClassifier variable
rfc = RandomForestClassifier()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
#   - 75% training and 25% test, seed random number generator with 30 for consistency across executions


# defining a grid search and parameters for finding optimal hyperparameters
parameters = {
    'n_estimators': [100], # 50, 150, 200, 500          - values included in initial grid search
    'max_features': ['log2'], # 'log2', 'sqrt'
    'criterion': ['entropy'], # 'gini', 'log_loss'
    'random_state': [30] # 20, 25, 35, 40, 42
}
cv_rfc = GridSearchCV(estimator=rfc, param_grid=parameters, n_jobs=-1, cv=10) # estimator=rfc removed for pipe testing
cv_rfc.fit(X_train, y_train) # .values.ravel()

# applying model to testing data
print("Applying model to testing data")
y_pred = cv_rfc.predict(X_test)

# generating F1 score and training accuracy score
print("F1 Score: " + str(f1_score(y_test, y_pred)))
print("Training Accuracy: " + str(cv_rfc.score(X_test, y_test)))

# applying the model to the unseen test data
print("Applying Model to unseen data")
predLabels = cv_rfc.predict(testingData)
print(predLabels)

# output the results to a csv file
outputData = testingData
outputData[128] = predLabels
outputData.to_csv("TestingResultsBinary.csv", header=False, index=False)

# generating confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=cv_rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cv_rfc.classes_)
disp.plot()
plt.show()
