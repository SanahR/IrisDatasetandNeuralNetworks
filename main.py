from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
#Out of all the imports added, Perceptron, random, numpy, and matplotlib weren't used in this specific cell. 

#Creating the dictionary of hyper-parameters I want the GridSearchCV to try. 
parameters =  {"hidden_layer_sizes":[(1,),(1,1),(2,2),(5,5),(10,5),(15,5),(20,3)],"max_iter":[5000,7000,10000],"learning_rate_init":[0.1,0.01,0.001,0.0001]}
#####################
# CREATING THE DATA #
#####################
iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.3,shuffle = True)

##############################
# CREATING/TRAINING THE MODEL#
##############################

#Creating the model based on the best parameters from previous experimentation
#Please note that model isn't actually needed, as y_pred could be created with clf.predict(x_test). I chose to do it this way because I prefer it, but it will run without creating the model separately. 
model = MLP(hidden_layer_sizes = (5,5),max_iter = 7000,learning_rate_init = 0.01,activation = 'logistic')
model.fit(x_train,y_train)

#Creating the GridSearchCV
clf = GridSearchCV(MLP(), param_grid = parameters,verbose = 1)
clf.fit(x_train,y_train)

#Printing out key information from the cross-validation

print(clf.cv_results_.keys())
#Best parameters were typically a (5,5) hidden_layer_sizes, 7000 maximum iterations, and between 0.001 and 0.1 learning_rate_init
print(clf.best_params_)
#Best scores were typically in the range of 98.5 percent
print(clf.best_score_)

##############################
# PREDICTION AND TESTING     #
##############################
y_pred = model.predict(x_test)
#The average accuracy of this algorithm based on 20 runs is approximately 97.78 percent. 

#Multiplying the accuracy score by 100 to get a percentage, then rounding that to clean the number up. 
percentage = round((accuracy_score(y_test,y_pred)*100),2)
print("The accuracy score of this algorithm was, ",percentage)

