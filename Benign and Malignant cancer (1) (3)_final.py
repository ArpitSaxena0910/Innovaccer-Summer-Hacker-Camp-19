
# coding: utf-8

# # Cancer Detection
# ___
# 
# - Class
# 
#     **2** : `Benign Cancer`
#     
#     **4** : `Malignant Cancer`
#     
# 

# ### Importing Libraries

# In[59]:


import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ### Loading Data

# In[60]:


data = pd.read_excel(r"C:\Users\arpit\Desktop\Inter\data_second.xlsx")
data.head()


# In[61]:


del(data[' 4 for malignant)'])


# In[62]:


data.dtypes


# In[63]:


data.describe()


# ### Data Preprocessing

# In[64]:


## Labelling Malignant Cancer as 1 (False cases benign cancer as 0)

data['Class: (2 for benign,  4 for malignant)'].loc[data['Class: (2 for benign,  4 for malignant)'] == 2,] = 0
data['Class: (2 for benign,  4 for malignant)'].loc[data['Class: (2 for benign,  4 for malignant)'] == 4,]= 1


# In[65]:


data.head()


# In[66]:


# Distribution of class labels
print (data['Class: (2 for benign,  4 for malignant)'].value_counts())


# In[67]:


## null cases
pd.isnull(data).sum()


# In[68]:


data.columns


# In[69]:


# df = data[['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
#        'Uniformity of Cell Shape', 'Marginal Adhesion',
#        'Single Epithelial Cell Size', 'Bare_Nuclei', 'Bland Chromatin',
#        'Normal Nucleoli', 'Mitoses', 'Class: (2 for benign,  4 for malignant)']]

df = data


# In[70]:


df.head()


# In[71]:


df['Bare_Nuclei'].unique()


# In[72]:


a = df['Bare_Nuclei'][df['Bare_Nuclei'] != '?']

print (a.value_counts())
print (a.mean())
print (a.median())


# In[73]:


## Treating Missing values
## As frequency of 1 is much higher than others we prefer to go with mode = 1 to fill missing values

df['Bare_Nuclei'].loc[df['Bare_Nuclei']=='?',] = 1


# In[74]:


df.dtypes


# In[75]:


## distribution of class label
df.hist(column = 'Class: (2 for benign,  4 for malignant)')


# In[76]:


list(df.columns)[1:10]


# In[77]:


## distribution of features..
col_list = list(df.columns)[1:10]
for col in col_list:
    df.hist(column = col, bins=20)


# In[78]:


df.columns


# In[79]:


feature_list = ['Clump Thickness', 'Uniformity of Cell Size',
       'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare_Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses']


# ## Data Modelling
# ___

# ### K-Fold Logistic regression

# In[99]:


from sklearn.model_selection import KFold
X = df[feature_list].values
Y = df['Class: (2 for benign,  4 for malignant)'].values
kf = KFold(n_splits=5)


# In[81]:


from sklearn.model_selection import cross_val_score,train_test_split
logreg = LogisticRegression()
lr_cv_score = cross_val_score(logreg, X, Y, cv=5, scoring='roc_auc')
print("=== All AUC Scores ===")
print(lr_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Logistic Regression: ", lr_cv_score.mean())


# In[82]:


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


# In[83]:


from sklearn.model_selection import KFold
X = df[feature_list].values
Y = df['Class: (2 for benign,  4 for malignant)'].values
kf = KFold(n_splits=5)
kf.get_n_splits(X)


# In[84]:


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


# ### CV RandomForest

# In[85]:


from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier


X = df[feature_list]
y = df['Class: (2 for benign,  4 for malignant)'].values


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify = y)


# In[87]:


##  giving best results

# Max depth = 4
# n_estimators = 10
# Calss_weight = 'balanced'


# In[88]:


rfc = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=100, verbose=1, class_weight='balanced')
rfc_cv_score = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')


# In[89]:


print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


# In[90]:


print (rfc)


# In[91]:


rf = rfc.fit(X_train,y_train)
rf_predict = rf.predict(X_test)
rf_predict_proba = rf.predict_proba(X_test)


# In[92]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rf_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rf_predict))
print('\n')

tn, fp, fn, tp = confusion_matrix(y_test, rf_predict).ravel()
print("=== False Negatives {Actual : 1 (Malignant Cancer) Prediction : 0 (Benign Cancer)}  : ", fn)


# In[93]:


## Variable Importance

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print (importances)
importances.plot.bar()


# > Reducing False Negatives as we have to reduce cases where we predict benign cancer but actual is malignant cancer which represents False Negative here

# In[94]:


threshold = 0.25


# In[95]:


rf_predict_proba = rf.predict_proba(X_test)
rf_predict_t = rf_predict_proba[:,1]


# In[96]:


rf_predict_t[rf_predict_t < threshold] = 0
rf_predict_t[rf_predict_t >= threshold] = 1


# In[97]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rf_predict_t))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rf_predict_t))
print('\n')

tn, fp, fn, tp = confusion_matrix(y_test, rf_predict_t).ravel()
print("=== False Negatives {Actual : 1 (Malignant Cancer) Prediction : 0 (Benign Cancer)}  : ", fn)


# In[98]:


# As we decrease threshold from 0.5 to 0.2 False negatives goes from 3 to 0 and hence all malignant cases are predicted correctly 
# but we increase error in predicting benign cancer which tells us the tradeoff in selecting threshold. So, in order to reduce
# error in predicting malignant cancer we need to decrease threshold so that our model predicts class 1 with highest recall.  

