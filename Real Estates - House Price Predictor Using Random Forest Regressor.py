#!/usr/bin/env python
# coding: utf-8

# ## Real Estate - Price Predictor for Numerical Value

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("data.csv")


# In[3]:


dataset.head()


# In[4]:


dataset['CHAS'].value_counts()


# In[5]:


dataset.describe()


# ## Train-Test Splitting

# In[6]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[7]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['CHAS']):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


# In[8]:


strat_test_set['CHAS'].value_counts()


# In[9]:


strat_train_set['CHAS'].value_counts()


# In[10]:


dataset = strat_train_set.copy()


# ## Looking for Correlations

# In[11]:


corr_matrix = dataset.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[12]:


dataset.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying out Attribute combinations
# 

# In[13]:


dataset["TAXRM"] = dataset['TAX']/dataset['RM']


# In[14]:


dataset.head()


# In[15]:


corr_matrix = dataset.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[16]:


dataset.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[17]:


dataset = strat_train_set.drop("MEDV", axis=1)
dataset_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes
# To take care of missing attributes, we have three options:
#    1. Get rid of the missing data points
#    2. Get rid of the whole attribute
#    3. Set the value to some value(0, mean or median)
# In[18]:


a = dataset.dropna(subset=["RM"]) #Option 1
a.shape
# Note that the original dataset dataframe will remain unchanged


# In[19]:


dataset.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original dataset dataframe will remain unchanged


# In[20]:


median = dataset["RM"].median() # Compute median for Option 3


# In[21]:


dataset["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged


# In[22]:


dataset.shape


# In[23]:


dataset.describe() # before we started filling missing attributes


# In[24]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(dataset)


# In[25]:


imputer.statistics_


# In[26]:


X = imputer.transform(dataset)


# In[27]:


dataset_tr = pd.DataFrame(X, columns=dataset.columns)


# In[28]:


dataset_tr.describe()


# ## Scikit-learn Design

# Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters

# ## Creating a Pipeline

# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[30]:


dataset_num_tr = my_pipeline.fit_transform(dataset)


# In[31]:


dataset_num_tr.shape


# ## Using Random Forest Regressor Algorithm

# In[32]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor()
regression.fit(dataset_num_tr, dataset_labels)


# In[33]:


some_data = dataset.iloc[:5]


# In[34]:


some_labels = dataset_labels.iloc[:5]


# In[35]:


prepared_data = my_pipeline.transform(some_data)


# ## Evaluating the model

# In[36]:


import sklearn.metrics as matrics
from sklearn.metrics import mean_squared_error
dataset_predictions = regression.predict(dataset_num_tr)
mse = mean_squared_error(dataset_labels, dataset_predictions)
rmse = np.sqrt(mse)
mae = matrics.mean_absolute_error(dataset_labels, dataset_predictions)


# In[37]:


rmse


# In[38]:


mse


# In[39]:


mae


# ## Using better evaluation technique - Cross Validation

# In[40]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regression, dataset_num_tr, dataset_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[41]:


rmse_scores


# In[42]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[43]:


print_scores(rmse_scores)


# ## Saving the model

# In[44]:


from joblib import dump, load
dump(regression, 'regression.joblib') 


# ## Testing the model on test data

# In[45]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = regression.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_mae = matrics.mean_absolute_error(Y_test, final_predictions)


# In[46]:


final_rmse


# In[47]:


final_mse


# In[48]:


final_mae


# In[49]:


prepared_data[0]


# In[50]:


# It was our dataset output data
some_labels


# In[51]:


#Model Predicted This value. it was output value(MEDV)
regression.predict(prepared_data)


# ## Using the model

# In[52]:


from joblib import dump, load
import numpy as np
regression = load('regression.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
regression.predict(features)

