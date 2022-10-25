from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import math

 # LOAD IN THE DATASET
(full_X_train, full_y_train), (full_X_test, full_y_test) = mnist.load_data()

#use a smaller data set first (use half of the training data)
# full_X_train, dummyX, full_y_train, dummyY = train_test_split(full_X_train, full_y_train, test_size = 0.1)

X_train, y_train = [], []
X_test, y_test = [], []

# MODIFY THE DATA TO ONLY INCLUDE 3's and 8'set
for i in range(len(full_X_train)):
    if full_y_train[i] == 3 or full_y_train[i] == 8:
        X_train.append(full_X_train[i])
        y_train.append(full_y_train[i])

for i in range(len(full_X_test)):
    if full_y_test[i] == 3 or full_y_test[i] == 8:
        X_test.append(full_X_test[i])
        y_test.append(full_y_test[i])

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# PLOT THE TRAINING DATA

fig = plt.figure()
num_images = 20 # the number of images to display
grid_dim = math.ceil(num_images**0.5) # the number of images per row

for i in range(num_images):  
    plt.subplot(grid_dim,grid_dim,i+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()


# RESHAPE THE DATA
def make2D(X):
    nsamples, nx, ny = X.shape
    d2_X = X.reshape((nsamples,nx*ny))
    return d2_X

X_train = make2D(X_train)
X_test = make2D(X_test)



def makeKFolds(K, X, y):
    """
    Inputs:
        • K, the number of folds for this kfoldCV
        • X, nxp data matrix
        • y, nx1 response variable matrix
    
    Splits the data randomly into k folds. 

    Returns:
        • folds, a list of the form: 
            [(X_fold1, y_fold2), (X_fold2, y_fold2), .. , (X_foldk, y_foldk)]
    """

    folds = []
    for k in range(K, 1, -1):
        # randomly take 1/k of the data for this fold
        X, X_fold, y, y_fold = train_test_split(X, y, test_size = float(1)/float(k))
        folds.append((X_fold, y_fold))
    folds.append((X,y))
    return folds



    
def kFoldCV(K, X, y, X_test, y_test, model, param_vals):
    """
    Inputs:
        • K, the number of folds for this kfoldCV
        • X, nxp data matrix
        • y, nx1 response variable matrix
        • model, the model to use kFold CV on (1 parameter)
    
    Performs k-fold cross validation
    """
    # Keep track of the CV error's of each best lambda
    cv_errors = {}

    # Keep track of all lambdas and the corresponding errors in each fold
    all_cv_errors = defaultdict(list) # a mapping from each param_value to all the error values that it yields during CV
    all_training_errors = defaultdict(list)
    all_testing_errors = defaultdict(list)

     #1. Randomly split data into K-folds
    folds = makeKFolds(K, X, y)

    #progress bar for my sanity
    total = K*len(param_vals)
    current = 0
    
    #2. Parameter tuning
    for k, (X_val, y_val) in enumerate(folds):
        # the training set is the rest of the folds, merge them into one matrix
        remaining_folds = [folds[i] for i in range(len(folds)) if i!= k]
        X_train = np.concatenate([data[0] for data in remaining_folds], axis = 0)
        y_train = np.concatenate([data[1] for data in remaining_folds], axis = 0)
        
        # # standardize the data
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_train = scaler.transform(X_train)
        # X_val = scaler.transform(X_val)
        # X_test = scaler.transform(X_test)
       
         # fit the model for each parameter value, test error on validation set
        param_star, min_error = 0, float('inf') #keep track of the min error and corresponding param val
        for param_val in param_vals: 
            #print progress
            current +=1
            print(str(current/total*100)+"%")

            #fit the model for the parameter value
            model.set_params(C=param_val)
            model.fit(X_train, y_train)
            cv_error = log_loss(model.predict(X_val), y_val, labels = [3,8])

            if cv_error < min_error:
                param_star, min_error = param_val, cv_error
            all_cv_errors[param_val].append(cv_error) #store the errors for each param val to plot

            #compute train error
            train_error = log_loss(model.predict(X_train), y_train,labels = [3,8])
            all_training_errors[param_val].append(train_error)

            #compute test error
            test_error = log_loss(model.predict(X_test), y_test,labels = [3,8])
            all_testing_errors[param_val].append(test_error)

        # store the error of the chosen parameter for this fold
        cv_errors[param_star] = min_error
    
    #determine the optimal parameter
    final_param_star = min(cv_errors, key=cv_errors.get)

    avg_training_errors = [np.sum(errors)/len(errors) for errors in all_training_errors.values()]
    avg_testing_errors = [np.sum(errors)/len(errors) for errors in all_testing_errors.values()]
    # return the error data
    return all_cv_errors, avg_training_errors, avg_testing_errors, final_param_star


def plot_errors(K, param_vals, all_errors, training_errors, testing_errors):
    # error_dataframe = pd.DataFrame.from_dict(all_errors)
    # print(error_dataframe.head())

    # df.boxplot(by ='day', column =['total_bill'], grid = False)
    fig = plt.figure()

    #plot the average of all cv errors
    average_errors = [np.sum(errors)/len(errors) for errors in all_errors.values()]
    standard_devs = [np.std(errors)/len(errors) for errors in all_errors.values()]
    plt.plot(param_vals, average_errors, label = "avg cv errors", color='blue')

    #plot std lines
    plt.plot(param_vals, [average_errors[i]+standard_devs[i] for i in range(len(average_errors))], color='blue', linestyle='dashed')
    plt.plot(param_vals, [average_errors[i]-standard_devs[i] for i in range(len(average_errors))], color='blue', linestyle='dashed')

    #plot the testing and training errors
    plt.plot(param_vals, training_errors, label = "training error", color='red')
    plt.plot(param_vals, testing_errors, label = "testing error", color='green')


    #add labels
    plt.legend()
    plt.xlabel("1/lambdas (increasing model complexity)")
    plt.xscale("log")
    plt.ylabel("log loss")
    fig.savefig("errorplot.jpg")


# # Create the model
# clf = LinearSVC(penalty='l2', fit_intercept =True)
# clf = LogisticRegression(penalty='l2', solver='lbfgs', fit_intercept=True)
# K = 5

# param_vals = np.linspace(.1, 100, 5)

# # Use CV to train the model
# cv_errors, training_errors, testing_errors, final_param_star = kFoldCV(K, X_train, y_train, X_test, y_test, clf, param_vals)
# plot_errors(K, param_vals, cv_errors, training_errors, testing_errors)


# print(str(clf))
