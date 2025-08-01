# %% [markdown]
# # Model Training

# %% [markdown]
# ### scikit - learn
# 
# https://scikit-learn.org/stable/
# 
# scikitlearn (sklearn) provides simple and efficient tools for predictive data analysis. It is built on NumPy, SciPy, and matplotlib.

# %% [markdown]
# First thing, Import all the libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns', 50)

# %%
# next load the data
df = pd.read_csv('final.csv')
df.head()

# %%
df.tail()

# %%
df.shape

# %% [markdown]
# ## Linear Regression Model

# %%
# import linear regression model
from sklearn.linear_model import LinearRegression

# %%
# seperate input features in x
x = df.drop('price', axis=1)

# store the target variable in y
y = df['price']

# %% [markdown]
# **Train Test Split**
# * Training sets are used to fit and tune your models.
# * Test sets are put aside as "unseen" data to evaluate your models.
# * The `train_test_split()` function splits data into randomized subsets.

# %%
# import module
from sklearn.model_selection import train_test_split
df['property_type_Condo']= df['property_type_Condo'].astype(object)
df['property_tupe_Bunglow']=df['property_type_Bunglow'].astype(object)

# %%
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)

# %%
x_train.property_type_Bunglow.value_counts()

# %%
x_train.head()

# %%
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
# train your model
model = LinearRegression()
lrmodel = model.fit(x_train, y_train)

# %%
lrmodel.coef_

# %%
lrmodel.intercept_

# %%
x_train.head(1)

# %%
# make preditions on train set
train_pred = lrmodel.predict(x_train)

# %%
train_pred

# %%
# evaluate your model
# we need mean absolute error
from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(train_pred, y_train)
print('Train error is', train_mae)

# %%
lrmodel.coef_

# %%
lrmodel.intercept_

# %%
# make predictions om test set
ypred = lrmodel.predict(x_test)

#evaluate the model
test_mae = mean_absolute_error(ypred, y_test)
print('Test error is', test_mae)

# %% [markdown]
# ### Our model is still not good beacuse we need a model with Mean Absolute Error < $70,000
# 
# Note - We have not scaled the features and not tuned the model.

# %% [markdown]
#     

# %% [markdown]
# ## Decision Tree Model

# %%
# import decision tree model
from sklearn.tree import DecisionTreeRegressor

# %%
# create an instance of the class
dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)

# %%
# train the model
dtmodel = dt.fit(x_train,y_train)

# %%
# make predictions using the test set
ytrain_pred = dtmodel.predict(x_train)

# evaluate the model
train_mae = mean_absolute_error(ytrain_pred, y_train)
train_mae

# %%
# make predictions using the test set
ytest_pred = dtmodel.predict(x_test)

# %%
# evaluate the model
test_mae = mean_absolute_error(ytest_pred, y_test)
test_mae

# %% [markdown]
# ## How do I know if my model is Overfitting or Generalised?

# %%
# make predictions on train set
ytrain_pred = dtmodel.predict(x_train)

# %%
# import mean absolute error metric
from sklearn.metrics import mean_absolute_error

# evaluate the model
train_mae = mean_absolute_error(ytrain_pred, y_train)
train_mae

# %% [markdown]
#     

# %% [markdown]
# ## Plot the tree

# %%
# get the features
dtmodel.feature_names_in_

# %%
# plot the tree
from sklearn import tree

# Plot the tree with feature names
tree.plot_tree(dtmodel, feature_names=dtmodel.feature_names_in_)

#tree.plot_tree(dtmodel)
#plt.show(dpi=300)

# Save the plot to a file
plt.savefig('tree.png', dpi=300)

# %% [markdown]
#     

# %% [markdown]
#     

# %% [markdown]
#     

# %% [markdown]
# ## Random Forest Model

# %%
# import decision tree model
from sklearn.ensemble import RandomForestRegressor

# %%
# create an instance of the model
rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')

# %%
# train the model
rfmodel = rf.fit(x_train,y_train)

# %%
# make prediction on train set
ytrain_pred = rfmodel.predict(x_train)

# %%
# make predictions on the x_test values
ytest_pred = rfmodel.predict(x_test)

# %%
# evaluate the model
test_mae = mean_absolute_error(ytest_pred, y_test)
test_mae

# %%
# Individual Decision Trees
# tree.plot_tree(rfmodel.estimators_[2], feature_names=dtmodel.feature_names_in_)

# %% [markdown]
#     

# %% [markdown]
# ## Pickle:
# 
# * The pickle module implements a powerful algorithm for serializing and de-serializing a Python object structure.
# 
# * The saving of data is called Serialization, and loading the data is called De-serialization.
# 
# **Pickle** model provides the following functions:
# * **`pickle.dump`** to serialize an object hierarchy, you simply use `dump()`.
# * **`pickle.load`** to deserialize a data stream, you call the `loads()` function.

# %%
# import pickle to save model
import pickle

# Save the trained model on the drive
pickle.dump(dtmodel, open('RE_Model','wb'))

# %%
# Load the pickled model
RE_Model = pickle.load(open('RE_Model','rb'))

# %%
np.array(xtrain.loc[22])
ytrain[22]

# %%
# Use the loaded pickled model to make predictions
RE_Model.predict([[2012, 216, 74, 1 , 1, 618, 2000, 600, 1, 0, 0, 6, 0]])

# %%
x_test.head(1)

# %%



