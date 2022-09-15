import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import itertools
from sklearn.ensemble import RandomForestRegressor
from operator import itemgetter
import asgl
from sklearn.svm import SVR
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


#Load in and Process the Data
labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Read the data file
Data = pd.read_csv('./housing.csv.xls', header=None, delimiter=r"\s+", names=labels)

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
response = "MEDV"

# Visualize the data (get insight on feature distributions, correlations, etc)
fig, axs = plt.subplots(ncols = 7, nrows = 2, figsize = (20,10))
idx = 0
axs = axs.flatten()
for feature, vals in Data.items():
    sns.distplot(vals, ax = axs[idx])
    idx += 1
plt.tight_layout(pad = .4, w_pad = 0.5, h_pad = 5.0)
plt.savefig("data_distributions.jpg")

# plot the pairwise correlation of features
plt.figure(figsize=(20, 10))
sns.heatmap(Data.corr().abs(),  annot=True)
plt.savefig("data_pairwise_corr.jpg")


# Split the data into features/response and training/test
X, y= Data.loc[:,features], Data[response]


# plot the response before taking the log
fig = plt.figure(figsize=(20,10))
sns.distplot(y)
plt.tight_layout(pad = .4, w_pad = 0.5, h_pad = 5.0)
plt.savefig("response_dist.jpg")


y = np.log(y)  # take the log of the response since the distribution is slighly skewed

#plot the response after taking the log
fig = plt.figure(figsize=(20,10))
sns.distplot(y)
plt.tight_layout(pad = .4, w_pad = 0.5, h_pad = 5.0)
plt.savefig("response_dist_log.jpg")

# Split and standardize the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
list_numerical = X_train.columns
scaler = StandardScaler().fit(X_train[list_numerical])  #use just the training data to set the conditions for standardizing
X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])

# FEATURE SELECTION

# 1. Linear Regression pvalues
model = sm.OLS(y_train, sm.add_constant(X_train)).fit() 
print( str(model.pvalues.loc["const"]))
for feature in features:
    print(str(model.pvalues.loc[feature]))

# 2. Best Subsets
def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y_train,X_train[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X_train[list(feature_set)]) - y_train) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k):  
    results = []
    for combo in itertools.combinations(X_train.columns, k):
        results.append(processSubset(combo))
    
    models = pd.DataFrame(results)
    
    best_model = models.loc[models['RSS'].argmin()]
    # Return the best model
    return best_model
best_model = getBest(4)
print(getBest(4)["RSS"])
print(getBest(4)["model"].summary())



# 3. Recursive Feature Elimination
rfe = RFE(SVR(kernel="linear"), n_features_to_select=4, step = 1)
rfe.fit(X_train, y_train)
#print the results in sorted order based on their ranking
for x, y in (sorted(zip(rfe.ranking_ ,X_train.columns.to_list()), key=itemgetter(0))):
    print(x, y)


# 4. Lasso
clf = linear_model.Lasso(alpha=2).fit(X_train, y_train)
#print the results in sorted order based on their ranking
for x, y in (sorted(zip(clf.coef_, X_train.columns.to_list()), key = itemgetter(0))):
    print(round(x, 6), y)

# 5. Elastic net
regr = linear_model.ElasticNet(random_state = 0)
regr.fit(X_train, y_train)
#print the results in sorted order based on their ranking
for x, y in (sorted(zip(regr.coef_, X_train.columns.to_list()), key = itemgetter(0))):
    print(round(x, 6), y)


# 6. Adaptive Lasso
alpha = 0.1
gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)
n_samples, n_features = X_train.shape
weights =np.ones(n_features)
n_lasso_iterations = 5

for k in range(n_lasso_iterations):
    X_w = X_train / weights[np.newaxis, :]
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=True)
    clf.fit(X_w, y_train)
    coef_ = clf.coef_ / weights
    print("iteration: " + str(k))
    for x, y in (sorted(zip(coef_, X_train.columns.to_list()), key = itemgetter(0))):
        print(round(x, 6), y)
    weights = gprime(coef_)



# Regularization paths:
lambdas = np.linspace(.0001,1,400)
model = linear_model.Lasso(max_iter=1000, alpha = 0.001, fit_intercept = True)
coefs = []

for l in lambdas:
    model.set_params(alpha=l)
    model.fit(X_train, y_train)
    coefs.append(model.coef_)

ax = plt.gca()
ax.plot(lambdas, coefs, label = features)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.legend()
plt.ylabel('Coefficients (standardized)')
plt.title('Lasso Regularization Paths')
plt.show()














