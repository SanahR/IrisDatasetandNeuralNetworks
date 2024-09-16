# Neural Networks with the Iris Dataset
A short segment of code using the Iris dataset to test out neural networks. 
## The Process
When creating any kind of machine learning algorithm, I always divvy up my code into four simple segments. They are as follows: 
1. Data Creation
2. Creating the Algorithm(s)
3. Training the Algorithm(s)
4. Predicting/Testing
After following those steps, you can fill in the gaps with everything else that you need to add. So, here's how I created a simple, sweet algorithm with relatively high accuracies.

  First, I added in all of my imports as well as some default imports (such as numpy, matplotlib, and random) just in case I needed them. Since I was using GridSearchCV in this program for hyper-parameter optimization, I then had to create a standard Python dictionary featuring every aspect I wanted to test. 
  After the two pre-requisite steps, I was able to move on to the first step in my primary four-step process, data creation. To do this, I loaded up the data with Scikit learn before splitting it into 4 parts (x_train,y_train,x_test,and y_test) with train_test_split. 
  Once I had my data prepared, I created the GridSearchCV (or clf) by feeding it the parameter dictionary from earlier and MLP(). I also created the MLPClassifier model and gave it the parameters I knew worked well. Please note that creating "model" was completely unneeded on my part, but I wanted to do it. I also created a main.py file, which uses clf instead of creating something separate. I then used .fit on both clf(the GridSearchCV) and model(my MLPClassifier) to train them on x_train and y_train. 
  Finally, I used .predict to save the predictions to a variable called y_pred, which I combined with y_test to check the accuracy of the algorithm. I also printed out some key information from clf. 
## Issues I Ran Into
## The Results
## Statistics and Results
## Miscellaneous Facts
