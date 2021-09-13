#!/usr/bin/env python
# coding: utf-8

# ## Real Estate - Price Predictor for Numerical Value

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("data.csv")


# ## Train-Test Splitting

# In[3]:


from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[4]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['CHAS']):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


# In[5]:


strat_test_set['CHAS'].value_counts()


# In[6]:


strat_train_set['CHAS'].value_counts()


# In[7]:


dataset = strat_train_set.copy()


# ## Looking for Correlations

# In[8]:


corr_matrix = dataset.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[9]:


dataset.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying out Attribute combinations
# 

# In[10]:


dataset["TAXRM"] = dataset['TAX']/dataset['RM']


# In[11]:


dataset.head()


# In[12]:


corr_matrix = dataset.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[13]:


dataset.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[14]:


dataset = strat_train_set.drop("MEDV", axis=1)
dataset_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[15]:


a = dataset.dropna(subset=["RM"])
a.shape
# Note that the original dataset dataframe will remain unchanged


# In[16]:


dataset.drop("RM", axis=1).shape
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[17]:


median = dataset["RM"].median()


# In[18]:


dataset["RM"].fillna(median)


# In[19]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(dataset)


# In[20]:


imputer.statistics_


# In[21]:


X = imputer.transform(dataset)


# In[22]:


dataset_tr = pd.DataFrame(X, columns=dataset.columns)


# In[23]:


dataset_tr.describe()


# ## Creating Pipeline

# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[25]:


dataset_num_tr = my_pipeline.fit_transform(dataset)


# In[26]:


dataset_num_tr.shape


# ## Selecting a desired model for Real Estates

# In[27]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(dataset_num_tr, dataset_labels)


# In[28]:


some_data = dataset.iloc[:5]


# In[29]:


some_labels = dataset_labels.iloc[:5]


# In[30]:


prepared_data = my_pipeline.transform(some_data)


# In[31]:


regression.predict(prepared_data)


# In[32]:


list(some_labels)


# ## Evaluating the model

# In[33]:


import numpy as np
from sklearn.metrics import mean_squared_error
dataset_predictions = regression.predict(dataset_num_tr)
mse = mean_squared_error(dataset_labels, dataset_predictions)
rmse = np.sqrt(mse)


# In[34]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[35]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regression, dataset_num_tr, dataset_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[36]:


rmse_scores


# In[37]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[38]:


print_scores(rmse_scores)


# ## Saving the model

# In[39]:


from joblib import dump, load
dump(regression, 'regression.joblib') 


# ## Testing the model on test data

# In[40]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = regression.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))


# In[41]:


final_rmse


# In[42]:


prepared_data[0]


# ## Using the model

# In[43]:


from joblib import dump, load
import numpy as np
regression = load('regression.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
regression.predict(features)

