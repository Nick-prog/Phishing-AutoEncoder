import packages
import warnings
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

np.random.seed(15)
random.seed(15)

class Classifier:
    
    def __init__(self, x_train_pred, x_valid_pred, y_train, y_valid, test_feature, test_label):
      self.x_train_pred = x_train_pred
      self.x_valid_pred = x_valid_pred
      self.y_train = np.array(y_train)
      self.y_valid = y_valid
      self.x_test = test_feature
      self.y_test = test_label
      self.randomstate = 15

      # Set the warning filter globally to ignore the "X has feature names" warning
      warnings.simplefilter(action='ignore', category=UserWarning)
    
    def decision_tree(self, depth=5):
       '''
       Method to create, train, and predict on the DecisionTreeClassifier model.
       depth = max depth possible for decision tree classifier.
       '''
       print("\nDecisionTreeClassifier...", end='')
       dt = DecisionTreeClassifier(max_depth=depth, random_state=self.randomstate)
       dt.fit(self.x_train_pred, self.y_train.ravel())
       train_pred = dt.predict(self.x_valid_pred)
       test_pred = dt.predict(self.x_test)
       print("\nTraining Accuracy:", accuracy_score(self.y_valid, train_pred), "\nValidation Accuracy:", accuracy_score(self.y_test, test_pred))
       print("Precision: ", precision_score(self.y_valid, train_pred), "\nRecall: ", recall_score(self.y_valid, train_pred), "\nF1-Score: ", f1_score(self.y_valid, train_pred))

    def logistic_regression(self, max_iter=1000):
       '''
       Method to create, train, and predict on the LogisticRegression model.
       max_iter = max number of iterations for training
       '''
       print("\nLogicisticRegression...", end='')
       clf = LogisticRegression(max_iter=max_iter, random_state=self.randomstate)
       clf.fit(self.x_train_pred, self.y_train.ravel())
       score = clf.score(self.x_valid_pred, self.y_valid)
       train_pred = clf.predict(self.x_valid_pred)
       test_pred = clf.predict(self.x_test)
       print("\nTraining Accuracy:", accuracy_score(self.y_valid, train_pred), "\nValidation Accuracy:", accuracy_score(self.y_test, test_pred))
       print("Precision: ", precision_score(self.y_valid, train_pred), "\nRecall: ", recall_score(self.y_valid, train_pred), "\nF1-Score: ", f1_score(self.y_valid, train_pred))

    def support_vector_machine(self, kernel="linear", regularization=1):
       '''
       Method to create, train, and predict on the SVC model.
       kernel = parameter to transform data into a higher dimensional space.
       (linear, polynomial, radial basis function (RBF), and sigmoid)
       regularization = parameter to control the maximum margin and minimize
       the classification error. 
       '''
       print("\nSupport Vector Machine...", end='')
       svm = SVC(kernel=kernel, C=regularization, random_state=self.randomstate)
       svm.fit(self.x_train_pred, self.y_train.ravel())
       train_pred = svm.predict(self.x_valid_pred)
       test_pred = svm.predict(self.x_test)
       print("\nTraining Accuracy:", accuracy_score(self.y_valid, train_pred), "\nValidation Accuracy:", accuracy_score(self.y_test, test_pred))
       print("Precision: ", precision_score(self.y_valid, train_pred), "\nRecall: ", recall_score(self.y_valid, train_pred), "\nF1-Score: ", f1_score(self.y_valid, train_pred))

    def random_forest(self, estimators=100):
       '''
       Method to create, train, and predict on the RandomForest model.
       estimators = total number of estimators allowed for predictions.
       '''
       print("\nRandomForestClassifier...", end='')
       rf = RandomForestClassifier(n_estimators=estimators, random_state=self.randomstate)
       rf.fit(self.x_train_pred, self.y_train.ravel())
       train_pred = rf.predict(self.x_valid_pred)
       test_pred = rf.predict(self.x_test)
       print("\nTraining Accuracy:", accuracy_score(self.y_valid, train_pred), "\nValidation Accuracy:", accuracy_score(self.y_test, test_pred))
       print("Precision: ", precision_score(self.y_valid, train_pred), "\nRecall: ", recall_score(self.y_valid, train_pred), "\nF1-Score: ", f1_score(self.y_valid, train_pred))