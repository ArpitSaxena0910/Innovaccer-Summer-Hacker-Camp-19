
# coding: utf-8

# # TripAdvisor Score Detection
# 

# ## Importing Libraries

# In[2]:


import pandas
import numpy
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


# # Loading Data

# In[3]:


data = pandas.read_excel(r"C:\Users\arpit\Desktop\data.xlsx")
data.head()


# # Data Preprocessing

# In[4]:


data.describe()


# In[5]:


data.dtypes


# In[6]:


user_country = data['User_country']
count = user_country.value_counts()
print(count)


# In[7]:


# Creating one more column 'countries_new' which label all the countries with count less than 7 as 'Others'
# re-grouping of the User Country variable


# In[8]:


conditions = [data.User_country == 'USA',data.User_country == 'UK',
             data.User_country == 'Canada',data.User_country == 'Australia',
             data.User_country == 'Ireland',data.User_country == 'India',
             data.User_country == 'Mexico',data.User_country == 'Germany']
choices = ['USA', 'UK', 'Canada','Australia','Ireland','India','Mexico','Germany']
data['countries_new'] = numpy.select(conditions, choices, default='Others')


# In[9]:


pandas.isnull(data).sum()


# In[10]:


# One hot encoding of Categorical Features


# In[11]:


data_cat_feats = data[['Period of stay','Traveler type','Swimming Pool',
                      'Exercise Room','Basketball Court','Yoga Classes',
                      'Club','Free Wifi','Hotel name','Hotel stars',
                      'User continent','Review month','Review weekday',
                      'countries_new']]
data_one_hot = pandas.get_dummies(data_cat_feats, drop_first=True)


# In[12]:


# Numerical Features
data_con_feats = data[['Nr. reviews','Nr. hotel reviews','Helpful votes','Score','Nr. rooms','Member years']]

#Add all the features
data_cat_con_feats = pandas.concat([data_con_feats,data_one_hot], axis = 1)
data_cat_con_feats.columns


# In[13]:


data_cat_con_feats.dtypes


# In[14]:


## distribution of Score label
data.hist(column = 'Score')


# In[15]:


## distribution of Nr. reviews label
data.hist(column = 'Nr. reviews')


# In[16]:


## distribution of Nr. hotel reviews  label
data.hist(column = 'Nr. hotel reviews')


# In[17]:


## distribution of Helpful votes  label
data.hist(column = 'Helpful votes')


# In[18]:


## distribution of Nr. rooms  label
data.hist(column = 'Nr. rooms')


# In[19]:


## distribution of Nr. rooms  label
data.hist(column = 'Member years')


# In[20]:


## Treating Unusual values
# Removal of unusual data values (One value in Member Years column is negative)

data_cat_con_feats['Member years'][data_cat_con_feats['Member years']<0]=0


# In[21]:


feature_list = ['Nr. reviews',
 'Nr. hotel reviews',
 'Helpful votes',
 'Nr. rooms',
 'Member years',
 'Period of stay_Jun-Aug',
 'Period of stay_Mar-May',
 'Period of stay_Sep-Nov',
 'Traveler type_Couples',
 'Traveler type_Families',
 'Traveler type_Friends',
 'Traveler type_Solo',
 'Swimming Pool_YES',
 'Exercise Room_YES',
 'Basketball Court_YES',
 'Yoga Classes_YES',
 'Club_YES',
 'Free Wifi_YES',
 'Hotel name_Caesars Palace',
 'Hotel name_Circus Circus Hotel & Casino Las Vegas',
 'Hotel name_Encore at wynn Las Vegas',
 'Hotel name_Excalibur Hotel & Casino',
 'Hotel name_Hilton Grand Vacations at the Flamingo',
 'Hotel name_Hilton Grand Vacations on the Boulevard',
 "Hotel name_Marriott's Grand Chateau",
 'Hotel name_Monte Carlo Resort&Casino',
 'Hotel name_Paris Las Vegas',
 'Hotel name_The Cosmopolitan Las Vegas',
 'Hotel name_The Cromwell',
 'Hotel name_The Palazzo Resort Hotel Casino',
 'Hotel name_The Venetian Las Vegas Hotel',
 'Hotel name_The Westin las Vegas Hotel Casino & Spa',
 'Hotel name_Treasure Island- TI Hotel & Casino',
 'Hotel name_Tropicana Las Vegas - A Double Tree by Hilton Hotel',
 'Hotel name_Trump International Hotel Las Vegas',
 'Hotel name_Tuscany Las Vegas Suites & Casino',
 'Hotel name_Wyndham Grand Desert',
 'Hotel name_Wynn Las Vegas',
 'Hotel stars_4',
 'Hotel stars_5',
 'Hotel stars_3,5',
 'Hotel stars_4,5',
 'User continent_Asia',
 'User continent_Europe',
 'User continent_North America',
 'User continent_Oceania',
 'User continent_South America',
 'Review month_August',
 'Review month_December',
 'Review month_February',
 'Review month_January',
 'Review month_July',
 'Review month_June',
 'Review month_March',
 'Review month_May',
 'Review month_November',
 'Review month_October',
 'Review month_September',
 'Review weekday_Monday',
 'Review weekday_Saturday',
 'Review weekday_Sunday',
 'Review weekday_Thursday',
 'Review weekday_Tuesday',
 'Review weekday_Wednesday',
 'countries_new_Canada',
 'countries_new_Germany',
 'countries_new_India',
 'countries_new_Ireland',
 'countries_new_Mexico',
 'countries_new_Others',
 'countries_new_UK',
 'countries_new_USA']


# # Train test Split

# In[22]:


X = data_cat_con_feats[feature_list]
y = data_cat_con_feats['Score']


# In[23]:


from sklearn.model_selection import cross_val_score,train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# # Data Modelling

# # K Fold Linear Regression

# In[24]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X, y)


# In[25]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm, X,  y, cv=5,scoring='neg_mean_squared_error')
print(scores)
print((scores).mean())


# # K fold Random Forest

# In[29]:


from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor(n_estimators = 100,max_depth=5, min_samples_leaf= 4,min_samples_split= 8,random_state = 42,n_jobs = 5)
rfc_cv_score = cross_val_score(rf1, X, y, cv=5, scoring='neg_mean_squared_error')


# In[ ]:


rf1.fit(X_train, y_train)
rfi_predict=rf1.predict(X_test)


# In[30]:


print("=== All Mean Squared error Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean Mean Squared error Scores ===")
print("Mean Squared error Score - Random Forest: ", rfc_cv_score.mean())


# # Linear regression Results

# In[27]:


from sklearn.metrics import mean_squared_error, r2_score
lm.fit(X_train,y_train)
Y_predict = lm.predict(X_test)
Y_predict_train = lm.predict(X_train)
print("Mean squared error on testing data: %.2f"% mean_squared_error(y_test, Y_predict))
print("Mean squared error on training data: %.2f"% mean_squared_error(y_train, Y_predict_train))


# # Random Forest results

# In[36]:


rf1.fit(X_train, y_train)
rfi_predict=rf1.predict(X_test)
mse1 = mean_squared_error(y_test, rf1.predict(X_test))
print("MSE on test data: %.4f" % mse1)


# In[37]:


mse1 = mean_squared_error(y_train, rf1.predict(X_train))
print("MSE on train data: %.4f" % mse1)


# # Ensemble

# In[34]:


final_predict = (Y_predict + rfi_predict)/2


# In[38]:


mse1 = mean_squared_error(y_test, final_predict)
print("MSE on ensembled data: %.4f" % mse1)

