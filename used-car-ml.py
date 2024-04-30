#!/usr/bin/env python
# coding: utf-8

# # Import Data

# In[35]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


# import data
cars = pd.read_csv("data/Kia_Data2.csv", na_values="?").dropna()


# In[37]:


# print the first five lines
cars.head()


# In[38]:


# list basic info
cars.info()


# In[39]:


print(cars["model"].unique())


# In[40]:


# Find all the unique condition grades.
print(cars["condition_grade"].unique())

# Replace all the occurences of codes with the corresponding numbers.
cars.loc[cars["condition_grade"] == "AV", "condition_grade"] = '30'
cars.loc[cars["condition_grade"] == "SL", "condition_grade"] = '0'
cars.loc[cars["condition_grade"] == "CL", "condition_grade"] = '40'
cars.loc[cars["condition_grade"] == "EC", "condition_grade"] = '50'
cars.loc[cars["condition_grade"] == "PR", "condition_grade"] = '10'
cars.loc[cars["condition_grade"] == "RG", "condition_grade"] = '20'


# In[41]:


print(cars["condition_grade"].unique())


# In[42]:


print(cars["transmission"].unique())
cars.loc[cars['transmission'] == 'M', 'transmission'] = 1
cars.loc[cars['transmission'] == 'A', 'transmission'] = 0
cars.loc[cars['transmission'] == '5', 'transmission'] = 0
cars.loc[cars['transmission'] == '6', 'transmission'] = 0
cars.loc[cars['transmission'] == 'O', 'transmission'] = 0
cars.loc[cars['transmission'] == 'P', 'transmission'] = 0
cars.loc[cars['transmission'] == 'Z', 'transmission'] = 0
cars.loc[cars['transmission'] == 'C', 'transmission'] = 0
cars.loc[cars['transmission'] == 'N', 'transmission'] = 0


# In[43]:


print(cars["transmission"].unique())


# In[44]:


print(cars["season"].unique())


# In[45]:


date = []
for date in cars.sold_date[1:]:
    a_string = date
    first_word = a_string.split()[0]
    print(first_word)


# # Exploratory Data Analysis

# In[46]:


# convert object type to numberic values
cars.auction_code = pd.factorize(cars.auction_code)[0]
cars.color = pd.factorize(cars.color)[0]
cars.make = pd.factorize(cars.make)[0]
cars.model = pd.factorize(cars.model)[0]
cars.subseries = pd.factorize(cars.subseries)[0]
cars.body = pd.factorize(cars.body)[0]
cars.engine = pd.factorize(cars.engine)[0]
cars.sold_date = pd.factorize(cars.sold_date)[0]
cars.season = pd.factorize(cars.season)[0]


# In[47]:


print(cars["season"].unique())


# In[48]:


cars['model'].describe()


# In[49]:


cars['sale_price'].describe()


# In[50]:


sns.distplot(cars['sale_price']);


# In[51]:


#correlation matrix
corrmat = cars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,);


# In[52]:


# #saleprice correlation matrix
# k = 15 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'sale_price')['sale_price'].index
# cm = np.corrcoef(X[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()


# In[53]:


#scatterplot
sns.set()
cols = ['sale_price', 'car_year', 'model', 'mileage', 'condition_grade', 'season']
sns.pairplot(cars[cols], height = 2.5)
plt.show();


# In[54]:


#applying log transformation
#cars['mileage'] = np.log(cars['mileage'])
#cars['car_year'] = np.log(cars['car_year'])
#cars['sale_price'] = np.log(cars['sale_price'])


# In[55]:


# Create the x and y variables
X = cars.drop(['auction_code', 'color', 'make', 'subseries', 'body', 'engine', 'transmission', 'times_run', 'sold_date', 'seller', 'sale_price', 'sold_date_bad'], axis = 1)
selectedfeatures = X.columns
print(selectedfeatures)

y = cars['sale_price']


# In[56]:


#Create dummy variables
cars = pd.get_dummies(cars)


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)


# In[58]:


X_train.head()


# ## Linear regression model

# In[59]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Accuracy Score of Linear regression on train set",reg.score(X_train,y_train))
print("Accuracy Score of Linear regression on test set",reg.score(X_test,y_test))

#cars["reg_predict"] = reg.predict(X)


# In[60]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
results = sm.OLS(y_train, X_train.astype(float)).fit()
print(results.summary())


# In[61]:


# Root Mean Square Error (RMSE)
y_pred = reg.predict(X_test)
reg_mse = mean_squared_error(y_pred, y_test)
reg_rmse = np.sqrt(reg_mse)
print('Liner Regression RMSE: %.4f' % reg_rmse)


# In[62]:


# Mean absolute error (MAE)
lin_mae = mean_absolute_error(y_pred, y_test)
print('Liner Regression MAE: %.4f' % lin_mae)


# ## Decision tree model

# In[63]:


import pydotplus as pdp
from IPython.display import Image

from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor


regrtree = DecisionTreeRegressor(max_depth = 5)
regrtree.fit(X_train, y_train)
print("Accuracy Score of Decision Tree on train set",regrtree.score(X_train,y_train))
print("Accuracy Score of Decision Tree on test set",regrtree.score(X_test,y_test))

cars["tree_predict"] = regrtree.predict(X)


# In[64]:


y_pred_tree = regrtree.predict(X_test)
tree_mse = mean_squared_error(y_pred_tree, y_test)
tree_rmse = np.sqrt(tree_mse)
print('Decision Tree RMSE: %.4f' % tree_rmse)


# In[65]:


# Mean absolute error (MAE)
tree_mae = mean_absolute_error(y_pred_tree, y_test)
print('Decision Tree MAE: %.4f' % tree_mae)


# In[66]:


# Feature importance scores
print(list(zip(X_train, regrtree.feature_importances_)))

feat_importances = pd.Series(regrtree.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')


# In[67]:


from sklearn.externals.six import StringIO  
def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pdp.graph_from_dot_data(dot_data.getvalue())
    return(graph)


graph = print_tree(regrtree, features=selectedfeatures)
Image(graph.create_png())


# ## Random forest model

# In[77]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=0)
forest_reg.fit(X_train, y_train)
print("Accuracy Score of Random Forests on train set",forest_reg.score(X_train,y_train))
print("Accuracy Score of Random Forests on test set",forest_reg.score(X_test,y_test))

y_pred_forest = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_pred_forest, y_test)
forest_rmse = np.sqrt(forest_mse)
print('Random Forest RMSE: %.4f' % forest_rmse)

forest_mae = mean_absolute_error(y_pred_forest, y_test)
print('Random Forest MAE: %.4f' % forest_mae)

print("\nFeature Importances: ")
feature_names = X_train.columns
importances = forest_reg.feature_importances_
for name, importance in zip(feature_names, importances):
    print(name, "=", importance)


# # Boosting model
# 
# 

# In[70]:


from sklearn.ensemble import GradientBoostingRegressor
GBM = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
print("Accuracy equals %s" %(GBM.score(X_test,y_test)))


# In[71]:


from sklearn.ensemble import AdaBoostRegressor
ABM = AdaBoostRegressor(random_state=0).fit(X_train, y_train)
print("Accuracy equals %s" %(ABM.score(X_test,y_test)))


# In[72]:


# Feature importance scores
print(list(zip(X_train, forest_reg.feature_importances_)))

feat_importances = pd.Series(forest_reg.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')


# In[73]:


#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(max_depth=6)


# In[74]:


# Predict using test set.
#predictions = GBM.predict(X_test[X])

submission = pd.DataFrame({
        "Predicted": y_pred_forest,
        "sale_price": y_test,
        'car_year': X_test["car_year"], 
        'model': X_test["model"], 
        'mileage': X_test["mileage"],
        'condition_grade': X_test['condition_grade'],
        'season': X_test['season']    
})

submission.to_csv("data/Kia_Output.csv", index=False)


# # Conclusion
# 
# ### Based on this used car dataset I found from Kaggle, I was able to train several supervised machine learning models to predict the price of a used car. I started off with a simpler linear regression model, then a decision tree model, random forest model, and a gradient boosting model. 
# 
# ### I began the notebook by cleaning the data and performing EDA. The dataset had some inconsistent labels for the transmission type and for the condition of the car, so I transformed these into formats that would be more suitable for modeling. I performed EDA by viewing metadata and correlation matrices. 
# 
# ### Then I progressed to modeling. I started off with a simpler linear regression model, then a decision tree model, random forest model, and a gradient boosting model. By the accuracy score metric, the random forest model was the most accurate at predicting the used car sale price on the test data set. I evalutated feature importance for each model to determine the optimal feature set for model training.

# In[ ]:




