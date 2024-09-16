from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

parameters =  {"hidden_layer_sizes":[(1,),(1,1),(2,2),(5,5),(10,5),(15,5),(20,3)],"max_iter":[5000,7000,10000],"learning_rate_init":[0.1,0.01,0.001,0.0001]}
#####################
# CREATING THE DATA #
#####################
iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.3,shuffle = True)

##############################
# CREATING/TRAINING THE MODEL#
##############################
#model = MLP(hidden_layer_sizes = (5,5),max_iter = 7000,learning_rate_init = 0.01,activation = 'logistic')
#model.fit(x_train,y_train)
clf = GridSearchCV(MLP(), param_grid = parameters,verbose = 1)
clf.fit(x_train,y_train)

print(clf.cv_results_.keys())
print(clf.best_params_)
print(clf.best_score_)

##############################
# PREDICTION AND TESTING     #
##############################
y_pred = clf.predict(x_test)
percentage = round((accuracy_score(y_test,y_pred)*100),2)
print("The accuracy score of this algorithm was, ",percentage)
