
# coding: utf-8

# #  Types of supervised learning
#  Regression: Predict a continuous response
#  Classification: Predict a categorical response

# # Linear Regrssion on US Housing Price

# * load the Python libraries needed for this case study and run the code below to load the Boston housing dataset.

# # Import packages and dataset

# In[242]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# * We are going to use the Boston House Price dataset. 
# 
# * This dataset is used in machine learning and statistics by pretty much everyone.
# 
# * The dataset contains 506  Instances 14 of Attributes 
# 
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
# 

# In[243]:


# Load dataset
filename = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, delim_whitespace=True, names=names)


# * We are using pandas to load the data. 
# * We will also use pandas next to explore the data both with descriptive statistics and data visualization.

# 
# ![_auto_0](attachment:_auto_0)

# In[244]:


# Dimension and Descriptive statistics

dataset.shape


# In[245]:


dataset.head(10)


# In[246]:


# display the last 5 rows
dataset.tail()


# # Description
# **'describe()' method to get the statistical summary of the various features of the data set

# In[247]:


dataset.describe()


# **Correlation matrix 

# In[248]:


dataset.corr()


# # Feature and variable sets

# In[249]:


prices = dataset['MEDV']
features = dataset.drop('MEDV', axis = 1)


# In[250]:


prices.plot.hist(bins=25,figsize=(8,4))
plt.show()


# In[251]:


prices.plot.density(figsize=(4,4))
plt.show()


# In[252]:


plt.plot(dataset['RM'],prices,'.')
plt.show()


# # Developing a Model

# # 1. Prepare Data: Test-train split
# **Import train_test_split function from scikit-learn

# In[253]:


from sklearn.cross_validation import train_test_split
# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=validation_size, random_state=seed)


# ## Check the size and shape of train/test splits 

# In[254]:


print("Training feature set size:",X_train.shape)
print("Test feature set size:",X_test.shape)
print("Training variable set size:",Y_train.shape)
print("Test variable set size:",Y_test.shape)


# # Model fit and training
# ## Import linear regression model estimator from scikit-learn and instantiate

# In[255]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# # model coefficients

# In[256]:


print ("coefficients: %s\nintercept: %0.3f" % (regressor.coef_,regressor.intercept_))


# # Make Predictions
# # Model evaluation metrics for regression
# * Evaluation metrics for classification problems, such as accuracy, are not useful for regression problems. 
# * Instead, we need evaluation metrics designed for comparing continuous values.

# In[257]:


# make predictions on the testing set

y_pred = regressor.predict(X_test)


# * Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# * Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

# In[258]:


# Calculate (MSE) and root-mean-square error (RMSE)
# RMSE is interpretable in the "y" units.

from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error

lin_mse = mean_squared_error(y_pred, Y_test)
lin_rmse = np.sqrt(lin_mse)

print('Linear Regression MSE: %.4f' % lin_mse)
print('Linear Regression RMSE: %.4f' % lin_rmse)


# In[259]:


# function to calculate MAE; MSE and RMSE
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_test,y_pred))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_test,y_pred))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))


# In[260]:


# function to calculate r-squared

print ("R2 Squared : ", r2_score(y_pred, Y_test))


# In[261]:


plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted house prices",fontsize=25)
plt.xlabel("Actual test set house prices",fontsize=18)
plt.ylabel("Predicted house prices", fontsize=18)
plt.scatter(x=Y_test,y=y_pred)
plt.show()


# # STATMODEL

# In[262]:


import statsmodels.api as sm
# create a fitted model & print the summary
X_train = sm.add_constant(X_train)
lm = sm.OLS(Y_train, X_train).fit()
print (lm.summary())


# In[263]:


# print the R-squared value for the model
lm.rsquared


# In[264]:


# 1-Feature observation:    Ref (23p157 building ….
plt.plot(dataset['RM'],prices,'.')
plt.show()


# In[265]:


corr_matrix = dataset.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[266]:


RM = dataset['RM']
RM = np.transpose(np.atleast_2d(RM))


# In[267]:


from sklearn.metrics import r2_score
regress = LinearRegression()
regress.fit(RM, prices)

pred = regress.predict(RM)

print('RMSE on training, {:.2}'.format(np.sqrt(mean_squared_error(prices, pred))))
print('R2-squared on training, {:.2}'.format(r2_score(prices, pred)))
print('')


# In[268]:


# plotting fitted line
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.scatter(RM, prices, color='black')
plt.plot(RM, regress.predict(RM), color='blue', linewidth=3)
plt.title('prices vs RM')
plt.ylabel('prices')
plt.xlabel('RM')


# In[269]:


print ("coefficients: %s\nintercept: %0.3f" % (regress.coef_,regress.intercept_))


# In[270]:


X = dataset["RM"]
y = dataset["MEDV"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()


# In[317]:


X = dataset[["RM","LSTAT"]]
y = dataset["MEDV"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()


# # Interpreting the Output R-squared

#  We can see here that this model has a much higher R-squared value — 0.639, meaning that this model explains 64% of the variance in our dependent variable. Whenever we add variables to a regression model, R² will be higher. 
# This was the example of both single and multiple linear regression in Statsmodels. We could have used as little or as many variables we wanted in our regression model(s) — ( up to all the 13 attributes:  R-squared value = 0.770.

# In[318]:


# Split-out validation dataset
validation_size = 0.20
seed = 7
X_trai, X_tes, Y_trai, Y_tes = train_test_split(X, y, test_size=validation_size, random_state=seed)


# In[319]:


print("Training feature set size:",X_trai.shape)
print("Test feature set size:",X_tes.shape)
print("Training variable set size:",Y_trai.shape)
print("Test variable set size:",Y_tes.shape)


# In[320]:


regr = LinearRegression()
regr.fit(X_trai, Y_trai)


# In[322]:


print ("coefficients: %s\nintercept: %0.3f" % (regr.coef_,regr.intercept_))


# In[323]:


# make predictions on the testing set

y_predic = regr.predict(X_tes) 


# In[324]:


# function to calculate r-squared

print ("R2 Squared : ", r2_score(y_predic, Y_tes))

