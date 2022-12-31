#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction by using ML

# In[113]:


#Import all the liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for warnings
import warnings
warnings.filterwarnings('ignore')


# In[114]:


#Load the dataset
df=pd.read_csv("C:\\Users\\ASUS\\Desktop\\DATA\\framingham.csv")
df


# In[115]:


df.shape        #total no. of rows and columns


# In[116]:


df.head()       #first 5 rows


# In[117]:


df.tail()      #last 5 rows


# In[118]:


df.info()         #Information of the data


# In[119]:


df.describe()


# In[120]:


df.isnull().sum()     #check null values


# In[121]:


#Education column has no relation with Heart Disease.

#So drop education column.


# In[122]:


df1=df.drop(columns=["education"])
df1


# In[123]:


#show the correlation matrix
df1.corr()


# In[124]:


#Draw the map to show the correlation graphically.
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.heatmap(df1.corr(),annot=True,linewidths=2)


# In[125]:


#currentsmoker and cigperday have high correlation
#sysbp and diabp have high correlation.

#Hence select one from each pair.


# In[126]:


df2=df1.drop(columns=['currentSmoker','diaBP'])
df2.head()


# In[127]:


df2.shape


# In[128]:


#Handle the missing values, outliers and Duplicate data.


# In[129]:


df2.isnull().sum()


# In[130]:


#Handle the missing values.
#Replace the missing values by the median value.


# In[131]:


df2['BPMeds']=df2['BPMeds'].replace(np.nan,df2['BPMeds'].median())


# In[132]:


df2['totChol']=df2['totChol'].replace(np.nan,df2['totChol'].median())


# In[133]:


df2['BMI']=df2['BMI'].replace(np.nan,df2['BMI'].median())


# In[134]:


df2['heartRate']=df2['heartRate'].replace(np.nan,df2['heartRate'].median())


# In[135]:


df2['glucose']=df2['glucose'].replace(np.nan,df2['glucose'].median())


# In[136]:


df2['cigsPerDay']=df2['cigsPerDay'].replace(np.nan,df2['cigsPerDay'].median())


# In[137]:


df2.isnull().sum()


# In[138]:


#Outlier detection and removal

#outlier-:It ia a value that exceeds the normal range.


# In[139]:


df2[df2['totChol']>450]


# In[140]:


df2[df2['sysBP']>220]


# In[141]:


len(df2[df2['cigsPerDay']>50])


# In[142]:


len(df2[df2['BMI']>45])


# In[143]:


len(df2[df2['heartRate']>125])


# In[144]:


len(df2[df2['glucose']>200])


# In[145]:


#Remove outliers


# In[146]:


df2=df2[~(df2['totChol']>450)]


# In[147]:


df2=df2[~(df2['sysBP']>220)]


# In[148]:


df2=df2[~(df2['BMI']>45)]


# In[149]:


df2=df2[~(df2['heartRate']>125)]


# In[150]:


df2=df2[~(df2['glucose']>200)]


# In[151]:


df2.shape


# In[152]:


#Find Duplicates
len(df2[df2.duplicated()])


# In[153]:


#So our dataset has no duplicate values.


# In[154]:


#Separate 
X = df2.drop(columns=['TenYearCHD'])
X


# In[174]:


Y = df2['TenYearCHD']
Y


# In[175]:


# SCALING THE DATA

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[176]:


X_train


# In[177]:


X_test


# In[187]:


#Create LInear regression model
from sklearn.linear_model import LinearRegression
L=LinearRegression()
#Train the model
L.fit(X,Y)


# In[188]:


#Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=9)


# In[189]:


X_train.shape


# In[190]:


X_test.shape


# In[191]:


#test the model
Y_pred=L.predict(X_test)
Y_pred


# In[192]:


Y_test.values


# In[193]:


#Find accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)
acc_LR=round(acc*100,2)
acc_LR


# In[186]:


#Draw the confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

     PN    PY
    
AN  1065   3

AY  187    2
# # Support Vector Machine

# In[182]:


#Create the model
from sklearn.svm import SVC
model= SVC()


# In[183]:


#Train the model
model.fit(X_train,Y_train)


# In[184]:


#Test the model
Y_pred=model.predict(X_test)
(Y_pred)


# In[185]:


#Find Accuracy
acc_svm= accuracy_score(Y_test,Y_pred)
acc_svm=round(acc_svm*100,2)
print("Accuracy of the model in SVM: ",acc_svm)


# # K-Nearest Neighbors(KNN)

# In[186]:


#Create the model
from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier()


# In[187]:


#Train the model
K.fit(X_train,Y_train)


# In[188]:


#Test the model
Y_pred=K.predict(X_test)
Y_pred


# In[189]:


Y_test.values


# In[190]:


#Find accuracy
acc_knn=accuracy_score(Y_test,Y_pred)
acc_knn=round(acc_knn*100,2)
print("Accuracy of the model in KNN:", acc_knn)


# In[191]:


#Display confusion matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

     PN     PY
    
AN   1033   35

AY   177    12
# # Decision Tree

# In[192]:


#Implement Decision Tree Calssifier
from sklearn.tree import DecisionTreeClassifier
D=DecisionTreeClassifier(criterion='gini',random_state=10)


# In[193]:


#Train the model
D.fit(X_train,Y_train)


# In[194]:


#Test the model
Y_pred=D.predict(X_test)
Y_pred


# In[195]:


#Find accuracy
acc_DTC=accuracy_score(Y_test,Y_pred)
acc_DTC=round(acc_DTC*100,2)
print("Accuracy of the model in DTC:", acc_DTC)


# In[196]:


#Display confusion matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

     PN     PY
    
AN   894    174

AY   138    51
# # Random Forest

# In[200]:


#Implement Random Forest 
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()


# In[202]:


#Train the model
model.fit(X_train,Y_train)


# In[203]:


Y_pred= model.predict(X_test)


# In[204]:


Y_pred


# In[205]:


Y_test


# In[206]:


#Find accuracy
acc_rf=accuracy_score(Y_test,Y_pred)
acc_rf=round(acc_rf*100,2)
print("Accuracy of the model in RF:", acc_rf)


# In[207]:


#Display confusion matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

     PN     PY
    
AN   1062   6

AY   176    13
# In[208]:


# Evaluating Models

model_eval = pd.DataFrame({'Model': ['Logistic Regression','K-Nearest Neighbour','SGD',
                                   'SVM','Decision Tree','Random Forrest'], 
                         'Accuracy': [acc_LR*100, acc_KNN*100, 
                                      acc_SGD*100,acc_SV*100,acc_DT*100,acc_RF*100]})
model_eval


# In[ ]:




