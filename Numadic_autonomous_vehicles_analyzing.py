#!/usr/bin/env python
# coding: utf-8

# <h1>Problem Statement</h1>
# 
# <p>The objective of this project is to develop a machine learning model that accurately predicts the manufacturer's suggested retail price (Base MSRP) of vehicles based on their features. By leveraging regression algorithms and utilizing the available dataset, we aim to estimate the price of vehicles in a way that benefits both consumers and manufacturers.</p>

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# ### Reading the Data from file

# In[2]:


df=pd.read_csv('EV_Registration_Dataset.csv')
df.head()


# In[3]:


df.info()


# In[4]:


# Datatypes are matching correctly with the domain.


# In[5]:


df.drop(columns=['Identifier'],inplace=True)


# In[6]:


# Identifier is not a significant feature for my analysis


# In[7]:


df.head()


# ### Checking Data

# <b>Missing values</b>

# In[8]:


df.isnull().sum()/len(df)*100


# <p> The 'Model' and 'Legislative District' columns contain a higher number of missing values. </p>

# In[9]:


# Checking the Model to identify missing values.

df_filter_model = df[df['Model'].isna()] 
result = df_filter_model.groupby(['Make','Electric Vehicle Type']).size().reset_index(name='count')
result


# <p> In the dataset, 'VOLVO' is listed as the 'Make,' and 'Battery Electric Vehicle (BEV)' is categorized under the 'Electric Vehicle Type,' both of which have missing values in the 'Model' column.</p>

# In[10]:


df_filter_make = df[df['Make']== 'VOLVO' ] 
df_filter_make = df_filter_make[df_filter_make['Electric Vehicle Type']== 'Battery Electric Vehicle (BEV)'] 
result = df_filter_make.groupby(['Model','Electric Vehicle Type']).size().reset_index(name='count')
result


#  <p> 'XC40' is the most frequently occurring entry under the 'Model' column for 'VOLVO' in the 'Make' category. Therefore, we are assigning 'XC40' to all the missing 'Model' entries. </p>

# In[11]:


df['Model'].fillna('XC40', inplace=True)


# <p> Checking the 'Legislative District' to identify missing values.</p>

# In[12]:


df_filter_lg = df[df['Legislative District'].isna()] 
result = df_filter_lg.groupby('City').size().reset_index(name='count') 
result


# <p> Given that the 'Legislative District' column has only a minimal number of missing values when analyzed on a 'City' column basis, we can proceed to remove or drop these entries.</p>

# <p>Since the number of missing values is minimal for other columns, we can safely remove these data points without significantly affecting our analysis.</p>

# In[13]:


df.dropna(inplace=True)
df.isna().sum()


# <p> Extract numerical values for longitude and latitude from the 'Vehicle Location' column.</p>

# In[14]:


# Utilize regular expressions (regex) to extract numerical values from the 'Vehicle Location' column
reg_expression = r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)';
df[['Longitude', 'Latitude']] = df['Vehicle Location'].str.extract(reg_expression)
df.head(2)


# In[15]:


# Convert 'Longitude' and 'Longitude' to numeric
df['Longitude'] = pd.to_numeric(df['Longitude'])
df['Latitude'] = pd.to_numeric(df['Latitude'])


# In[16]:


# Remove the 'Vehicle Location' column from the dataset since its values have been converted into the 'Longitude' and 'Latitude' columns.
df = df.drop(columns = 'Vehicle Location')


# In[17]:


df.head(2)


# In[18]:


df.isnull().sum()/len(df)*100


# #### Duplicate value in Dataset

# In[19]:


df.duplicated().sum()


# <p> Duplicate value is not present in Dataset.</p>

# #### Remove Outliers in Data Set

# In[20]:


df.head()


# In[21]:


con_cols = ['Postal Code','Model Year','Electric Range','Base MSRP','Legislative District','2020 Census Tract','Longitude','Latitude']
cat_cols = ['City','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Electric Utility']


# In[22]:


for col in con_cols:
    sns.boxplot(x=col,data=df)
    plt.show()


# <p> The presence of extreme outliers in the 'Postal Code,' 'Model Year,' 'Longitude,' and 'Latitude' columns leads us to ignore  outliers.</p>

# <p>Most of the values in 'Base MSRP' are zero, so we have decided not to treat these as outliers. </p>

# In[23]:


print("Base MSRP Values: ",df['Base MSRP'].unique())
print("*"*100)
print("Base MSRP is Zero: ", np.sum(df['Base MSRP'] == 0))
print("Base MSRP is not Zero: ", np.sum(df['Base MSRP'] != 0))


# <p> Since the majority of values in the 'Base MSRP' column are zero, and our target variable is also 'Base MSRP,' we should remove all rows where 'Base MSRP' equals zero.</p>

# In[24]:


df = df[df['Base MSRP'] != 0]
print("Base MSRP is Zero: ", np.sum(df['Base MSRP'] == 0))


# In[25]:


# Flooring and Capping for removing outliers 
columns = ['2020 Census Tract']
for column in columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - (1.5 * iqr)
    ul = q3 + (1.5 * iqr)
    for index in df[column].index:
        if df.loc[index,column]>ul:
            df.loc[index,column]=ul
        if df.loc[index,column]<ll:
            df.loc[index,column]=ll

# After removing outliers from '2020 Census Tract'
for column in columns:
    sns.boxplot(x=column,data=df)
    plt.show()


# <h1>EDA</h1>

# In[26]:


print(con_cols)
print(cat_cols )


# <b> Univariate Analysis <b>

# In[27]:


# Continuous Data - Histogram or KDE


# In[28]:


for col in con_cols:
    sns.kdeplot(x=col,data=df)
    plt.show()


# In[29]:


for col in con_cols:
    print(col,":",df[col].skew())


# <p>negatively skewed - Latitude
#     
# positively skewed - Postal Code, Base MSRP, Longitude
#     
# symmetric skewed - Model Year, Electric Range, Legislative District, 2020 Census Tract 
#  </p>

# In[30]:


# Normalize skewed variables in Python


# In[31]:


df['Base MSRP'].value_counts(normalize=True).round(2)


# In[32]:


# positively skewed - Postal Code, Base MSRP, Longitude

# Square Root Transformation:
pos_skewed_col = ['Postal Code', 'Base MSRP']
for col in pos_skewed_col:
    df[col] = np.sqrt(df[col])


# In[33]:


# negatively skewed - Latitude
# Square Transformation:
neg_skewed_col = ['Latitude']
for col in neg_skewed_col:
    df[col] = np.square(df[col])


# In[34]:


df.head()


# In[ ]:





# In[35]:


# Categorical Data - Bar graphs


# In[36]:



for col in cat_cols:
    plt.figure(figsize=(20,8))
    sns.countplot(x=col,data=df)
    plt.show()


# In[37]:


# Inferences:

# 1) "MODEL S" has higer number of count
# 2) "Bettery Electric Vehicke (BEV)" are morethan other Electri Vehile
# 3) Mostly "Clean Alternative Fuel Vehicle (CAFV)  are eiligible. 


# <b> Bivariate Analysis <b>

# In[38]:



# Target vs Continuous data

# categorical vs continuous data

# box plots


# In[39]:


for col in con_cols:
    sns.boxplot(x='Base MSRP',y=col,data=df)
    plt.show()


# In[40]:


# Target vs Categorical data
# Categorical vs Categorical

# Stacked bar chart


# In[41]:


for col in cat_cols:
    pd.crosstab(df[col],df['Base MSRP']).plot(kind='bar',figsize=(20,10))
    plt.show()


# <b> Correlation </b>

# In[42]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,fmt="0.2f") 


# <p> How Target variable is related with IDV?
# 
# 1) 'Model Year' has 57% negative correlation
# 
# 2) 'Electric Range' has 52% positive correlation
# </p>

# ## Scaling

# In[43]:


# Scaling  - Continuous data


# In[44]:


from sklearn.preprocessing import StandardScaler


# In[48]:


print(con_cols)


# In[50]:


con_cols.remove('Base MSRP')


# In[51]:


ss = StandardScaler()
X_con_scaled = pd.DataFrame(ss.fit_transform(df[con_cols]),columns=con_cols,index=df.index)
X_con_scaled.head()


# In[65]:


print("X_con_scaled: ", X_con_scaled.shape)


# ## Encoding

# In[52]:


# Categorical data - Numerical Data
# one hot encoding


# In[53]:


print(cat_cols)


# In[54]:


X_cat_enc = pd.get_dummies(df[cat_cols],drop_first=True) # function to execute one hot encoding


# #### Merge Cat and Con data

# In[101]:


X_final = pd.concat([X_con_scaled,X_cat_enc],axis=1)
X_final.head()


# In[102]:


X_final.shape


# # Train Test Split

# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


y= df['Base MSRP']


# In[105]:


# Training - 80%, Testing = 20% (Random selection of train_test_split)


# In[106]:


X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size=0.2,random_state=42)


# In[107]:


print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# In[108]:


print("Tarin: " ,round(3419*0.8))
print("Test: " ,round(3419*0.2))


# In[ ]:





# <p> <b> Linear Regression Model</b>:<br/>
#     
# Checking Linearity using Redidual Plot - Whether the data exhibits a linear relationship<br/>  </p> 
# 

# In[109]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
residuals = y_train - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# <p> The data does not exhibit linearity.</p>

# ## Impelementation Lasso Regression

# <b>Losso Regression</b>

# In[111]:


lasso = Lasso(alpha=1) #lamnda = alpha
lasso.fit(X_train,y_train)


print("Intercept:", lasso.intercept_)
print("Coefficient:", lasso.coef_)
 

print("*"*100)
print("\n")
print("Tain and Test Score")
#Tain and Test Score
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print('Train RMSE:',np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('Test RMSE:',np.sqrt(mean_squared_error(y_test, y_test_pred))) 

print('Train MAPE:',np.sqrt(mean_absolute_percentage_error(y_train, y_train_pred)))
print('Test MAPE:',np.sqrt(mean_absolute_percentage_error(y_test, y_test_pred)))

print('Train R2:',r2_score(y_train, y_train_pred))
print('Test R2:',r2_score(y_test, y_test_pred))

print("*"*100)
print("\n")
print("Cross Validation Score")

#Cross Validation Score

scores = cross_val_score(lasso,X_train,y_train,cv=3,scoring='r2') #cv=3 (train_split combinations)
print(scores)
print('Avg Score:', np.mean(scores))
print('Std Score: ', np.std(scores))


# <b> Hyperparameter Tuning</b>

# In[113]:


params = {'alpha':[0.1,0.5,0.8,1,1.2,1.5,2,2.5,3,3.5,4]}
# Lasso
grd_srch =GridSearchCV(Lasso(),params,scoring='r2',cv=3)
grd_srch.fit(X_train,y_train)
grid = pd.DataFrame(grd_srch.cv_results_)
grid


# In[ ]:





# ## Implementation of Decision Tree Classifier</b>

# In[114]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)


# In[115]:


from sklearn.tree import DecisionTreeClassifier


# In[116]:


dt_classfier = DecisionTreeClassifier(criterion='gini')
dt_classfier.fit(X_train,y_train)


# In[117]:


from sklearn import tree


# In[118]:


plt.figure(figsize=(20,20))
tree.plot_tree(dt_classfier)
plt.show()


# # Performance Metrics

# ### Train & Test Score

# In[119]:


from sklearn.metrics import confusion_matrix,classification_report,multilabel_confusion_matrix


# In[120]:


y_train_pred = dt_classfier.predict(X_train)
y_test_pred = dt_classfier.predict(X_test)


# In[121]:


label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)


# In[126]:


print(y_test.shape)
print(y_test_pred.shape)
print(y_train.shape)
print(y_train_pred.shape)


# In[127]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[128]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# In[129]:


# Full grown decision will always overfit


# ### Cross validation Score

# In[130]:


from sklearn.model_selection import cross_val_score


# In[202]:


scores = cross_val_score(dt_classfier,X_train,y_train,scoring='recall',cv=5,n_jobs = -1)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# # Hyperparameter Tuning for Decision Tree Classifier

# In[133]:


grid = {'max_depth':range(1,10),'min_samples_split':range(4,8,1),
       'max_leaf_nodes':range(3,10,1)}


# In[134]:


grid_src = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'),param_grid=grid,cv=5,
            scoring='recall')
grid_src.fit(X_train,y_train)


# In[135]:


pd.DataFrame(grid_src.cv_results_)


# In[136]:


grid_src.best_estimator_


# # Performance Metrics

# In[137]:


dt_tunned = DecisionTreeClassifier(criterion='gini',max_depth=3,max_leaf_nodes=8,
                                  min_samples_split=5,class_weight='balanced')
dt_tunned.fit(X_train,y_train)


# ### Train & Test Score

# In[138]:


y_train_pred = dt_tunned.predict(X_train)
y_test_pred = dt_tunned.predict(X_test)


# In[140]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[141]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# In[143]:


# Handled the overfitting and built and generalized model.


# ### Cross validation Score

# In[145]:


scores = cross_val_score(dt_tunned,X_train,y_train,scoring='recall',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# In[146]:


# Hyperparametert tuning is mandatory in Decision Tree. If we go with default values, we will get a 
# full grown Decision Tree which will overfit.


# # Feature Importance

# In[147]:


# Feature Importance will be in %


# In[149]:


fi = pd.DataFrame({"Columns":X_train.columns,
            "Feature_Importance": dt_tunned.feature_importances_
             }
            )
fi.sort_values(by='Feature_Importance',ascending=False)


# In[150]:


# Feature Importance are calculated by  gini importance


# # Ensemble Methods

# ## Bagging Classfier  

# In[151]:


from sklearn.ensemble import BaggingClassifier


# In[152]:


bag_class_log = BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'),
                             n_estimators=20,max_samples=0.8,max_features=0.5,
                              bootstrap=True,bootstrap_features=False
                         )
bag_class_log.fit(X_train,y_train)


# In[153]:


# To find the best parameters, perform Grid search


# ### Train & Test Score

# In[154]:


y_train_pred = bag_class_log.predict(X_train)
y_test_pred = bag_class_log.predict(X_test)


# In[155]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[156]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# ### Cross validation Score

# In[157]:


scores = cross_val_score(bag_class_log,X_train,y_train,scoring='recall',cv=5, )
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# # Implementation of Randomforest 

# In[158]:


from sklearn.ensemble import RandomForestClassifier


# In[159]:


rf = RandomForestClassifier(n_estimators=150,criterion='gini',max_depth=5,
                           min_impurity_decrease=4,max_leaf_nodes=8,max_samples=0.7,
                           max_features=0.5,bootstrap=True,class_weight='balanced')
rf.fit(X_train,y_train)


# ### Train & Test Score

# In[160]:


y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)


# In[161]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[162]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# ### Cross validation Score 

# In[163]:


scores = cross_val_score(rf,X_train,y_train,scoring='recall',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# In[164]:


# Model perfoermance is bad. Hyperparmeter tuning is needed


#  # Implementation of Gradient Boosting Classification

# In[165]:


from sklearn.ensemble import GradientBoostingClassifier


# In[166]:


gb_classifier = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=2)
gb_classifier.fit(X_train,y_train)


# ### Train & Test Score

# In[167]:


y_train_pred = gb_classifier.predict(X_train)
y_test_pred = gb_classifier.predict(X_test)


# In[168]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[169]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# ### Cross validation Score 

# In[170]:


scores = cross_val_score(gb_classifier,X_train,y_train,scoring='recall',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# In[173]:


# Since the data is high imbalanced. SMOTE is needed for the class balance
# Hyperparamater tuning is mandatory.


# # XGBoost

# In[ ]:


#!pip install xgboost


# In[175]:


import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)


X_train.columns = [regex.sub("_",col) if any(x in str(col) for x in set(('[',']','<'))) else col for col in X_train.columns.values] 
X_test.columns = [regex.sub("_",col) if any(x in str(col) for x in set(('[',']','<'))) else col for col in X_test.columns.values]


# In[176]:


from xgboost import XGBClassifier


# In[177]:


xgb = XGBClassifier(n_estimator=100, max_depth=3, reg_lambda=1,  learning_rate=0.1)
xgb.fit(X_train, y_train)


# In[178]:


xgb.predict(X_train)


# ### Train & Test Score

# In[179]:


y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)


# In[180]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[181]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# ### Cross validation Score 

# In[200]:


scores = cross_val_score(xgb,X_train,y_train,scoring='recall',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


#  # Bias & Variance Tradeoff

# In[183]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score


# In[186]:


# First way to do
depths = range(1,51)
train_scores = []
test_scores = []
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced')
    dt.fit(X_train, y_train)
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    train_scores.append(recall_score(y_train, y_train_pred, average='micro'))
    test_scores.append(recall_score(y_test, y_test_pred, average='micro'))


# In[187]:


plt.plot(depths, train_scores, color='red', marker='o')
plt.plot(depths, test_scores, color='blue', marker='*')
plt.show()


# In[195]:


# Second way to do
depths = range(1,51)
bias_scores = []
var_scores = []
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced')
    dt.fit(X_train, y_train)
    scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='recall')
    bias_scores.append(1-np.mean(scores))
    var_scores.append(np.std(scores)/np.mean(scores))
    
    


# In[ ]:





# In[196]:


plt.plot(depths, bias_scores, color='red', marker='o')
plt.plot(depths, var_scores, color='blue', marker='*')
plt.show()


# In[ ]:





# In[ ]:





# # Hypothesis Testing

# In[ ]:


import scipy.stats as stats


# In[ ]:


print(con_cols)


# <p><b>'Postal Code' has any impact on 'Base MSRP'.</b>
#     
# Null Hypothesis (H0): There is no significant impact of 'Postal Code' on 'Base MSRP'.
#     
# Alternative Hypothesis (H1): 'Postal Code' has a significant impact on 'Base MSRP'. </p>

# In[ ]:


postal_code_data = [data for data in df['Postal Code']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, postal_code_data)

alpha = 0.05 # Define the significance level (alpha)

# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Postal Code' has a significant impact on 'Base MSRP.'")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Postal Code' on 'Base MSRP.'")


# <p> <b>'Model Year' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Model Year' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Model Year' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


model_year_data = [data for data in df['Model Year']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, model_year_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Model Year' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Model Year' on 'Base MSRP'.")


# <p> <b>'Electric Range' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Electric Range' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Electric Range' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


electric_range_data = [data for data in df['Electric Range']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, electric_range_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Electric Range' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Electric Range' on 'Base MSRP'.")


# <p> <b>'Legislative District' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Legislative District' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Legislative District' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


legislative_district_data = [data for data in df['Legislative District']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, legislative_district_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Legislative District' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Legislative District' on 'Base MSRP'.")


# <p> <b>'2020 Census Tract' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Legislative District' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Legislative District' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


census_tract_data = [data for data in df['2020 Census Tract']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, census_tract_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): '2020 Census Tract' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of '2020 Census Tract' on 'Base MSRP'.")


# <p> <b>'Longitude' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Legislative District' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Legislative District' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


longitude_data = [data for data in df['Longitude']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, longitude_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Longitude' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Longitude' on 'Base MSRP'.")


# <p> <b>'Latitude' has any impact on 'Base MSRP'.</b>
# 
# Null Hypothesis (H0): There is no significant impact of 'Legislative District' on 'Base MSRP'.
# 
# Alternative Hypothesis (H1): 'Legislative District' has a significant impact on 'Base MSRP'.</p>

# In[ ]:


latitude_data = [data for data in df['Latitude']]  
base_msrp_data = [data for data in df['Base MSRP']]  
t_statistic, p_value = stats.ttest_ind(base_msrp_data, longitude_data)
# Define the significance level (alpha)
alpha = 0.05
# Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): 'Latitude' has a significant impact on 'Base MSRP'.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant impact of 'Latitude' on 'Base MSRP'.")


# <p> <b> Categorical variables has any impact on 'Base MSRP'.</b>
#     
# Null Hypothesis (H0): There is no significant association between the categorical variables and 'Base MSRP'.
# 
# Alternative Hypothesis (H1): There is a significant association between the categorical variables and 'Base MSRP'.</p>

# In[ ]:


# Create a contingency table from the DataFrame
#contingency_table = df[cat_col]

contingency_table = pd.crosstab(df['Base MSRP'], [df['City'], df['Make'],df[ 'Model'],df['Electric Vehicle Type'],
                                            df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'],df['Electric Utility']])
 
# Perform the chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the results
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies table:")
print(expected)

# Define the significance level (alpha)
alpha = 0.05

print("\n\n")

#Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): There is a significant association between the categorical variables and 'Base MSRP")
else:
    print("Fail to reject the null hypothesis (H0):  There is no significant association between the categorical variables and 'Base MSRP'.")


#  #### Apply chi-square tests to analyze association between categorical variables.

# In[ ]:


print(cat_cols)


# <b> Hypothesis for categorical variables</b>
# <p>Null Hypothesis (H0): There is no significant association between the categorical variables.
# 
# Alternative Hypothesis (H1): There is a significant association between the categorical variables.</p>

# In[ ]:


# Create a contingency table from the DataFrame
#contingency_table = df[cat_col]

contingency_table = pd.crosstab(df['City'],[df['Make'],df[ 'Model'],df['Electric Vehicle Type'],
                                            df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'],df['Electric Utility']])
 
# Perform the chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the results
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies table:")
print(expected)

# Define the significance level (alpha)
alpha = 0.05

print("\n\n")

#Check the p-value against the significance level
if p_value > alpha:
    print("Reject the null hypothesis (H0): There is a significant association between the categorical variables.")
else:
    print("Fail to reject the null hypothesis (H0):  There is no significant association between the categorical variables.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




