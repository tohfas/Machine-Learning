#!/usr/bin/env python
# coding: utf-8

# In[75]:


import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from scipy.stats import zscore
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

import os


# In[2]:


#Q1)1. Read the column description and ensure you understand each attribute well 
df = pd.read_csv("C:/Users/user/Downloads/Bank_Personal_Loan_Modelling.csv")
df_copy=df.copy()

df.head()


# In[3]:


df.info()


# In[4]:


df.isna().apply(pd.value_counts)   #null value check


# In[5]:


df.describe().T


# In[80]:


any(df['Experience'] < 0) #Replacing the negative values with the mean value of the column


# In[7]:


exp_med = df.loc[:,"Experience"].median()
df.loc[:, 'Experience'].replace([-1, -2, -3], [exp_med, exp_med, exp_med], inplace=True)


# In[8]:


any(df['Experience'] < 0)


# In[9]:


df.describe().T


# In[58]:


#Q2)Perform univariate analysis of each and every attribute - use an appropriate plot for a given attribute and
#mention your insights (5 points)


get_ipython().system('pip install pandas_profiling')
import pandas_profiling
df.profile_report()


# In[13]:


# Univariate Analysis of the continuous variables - 1

plt.figure(figsize= (40.5,40.5))
plt.subplot(5,3,1)
plt.hist(df.Age, color='orange', edgecolor = 'black')
plt.xlabel('Age')

plt.subplot(5,3,2)
plt.hist(df.Experience, color='blue', edgecolor = 'black')
plt.xlabel('Experience')

plt.subplot(5,3,3)
plt.hist(df.Income, color='brown', edgecolor = 'black')
plt.xlabel('Income')

plt.subplot(5,3,4)
plt.hist(df.CCAvg, color='lightblue', edgecolor = 'black')
plt.xlabel('Credit Card Average')

plt.subplot(5,3,5)
plt.hist(df.Mortgage, color='green', edgecolor = 'black')
plt.xlabel('Mortgage')

plt.show()



# OBSERVATION: Age and experience are distributed normally but Income Mortgage and CCAvg are highly left skewed


# In[12]:


# Checking for Skewness of data

import statsmodels.api as sm
import scipy.stats as stats
Skewness = pd.DataFrame({'Skewness' : [stats.skew(df.Age),stats.skew(df.Experience),stats.skew(df.Income),stats.skew(df.CCAvg)
                                      ,stats.skew(df.Mortgage)]},index=['Age','Experience','Income','CCAvg','Mortgage'])
Skewness


# In[15]:


#Univariate Analysis of the continuous variables2

plt.figure(figsize= (25,25))
plt.subplot(5,2,1)
sns.boxplot(x= df.Age, color='orange')

plt.subplot(5,2,2)
sns.boxplot(x= df.Experience, color='blue')

plt.subplot(5,2,3)
sns.boxplot(x= df.Income, color='brown')

plt.subplot(5,2,4)
sns.boxplot(x= df.CCAvg, color='lightblue')

plt.subplot(5,2,5)
sns.boxplot(x= df.Mortgage, color='green')


# OBSERVATION: Age is normalised with majority between 35 to 55 years
# Experience is also normalised  with majority between 11years to 30 yeaars
# Income is left skewed and is between 45K to 55K
# CCAvg and Mortgagae are again left skewed


# In[16]:


#Univariate Analysis of the categorical variables

plt.figure(figsize=(30,45))


plt.subplot(6,2,1)
df['Family'].value_counts().plot(kind="bar", align='center',color = 'blue',edgecolor = 'black')
plt.xlabel("Number of Family Members")
plt.ylabel("Count")
plt.title("Family Members Distribution")


plt.subplot(6,2,2)
df['Education'].value_counts().plot(kind="bar", align='center',color = 'pink',edgecolor = 'black')
plt.xlabel('Level of Education')
plt.ylabel('Count ')
plt.title('Education Distribution')


plt.subplot(6,2,3)
df['Securities Account'].value_counts().plot(kind="bar", align='center',color = 'violet',edgecolor = 'black')
plt.xlabel('Holding Securities Account')
plt.ylabel('Count')
plt.title('Securities Account Distribution')


plt.subplot(6,2,4)
df['CD Account'].value_counts().plot(kind="bar", align='center',color = 'yellow',edgecolor = 'black')
plt.xlabel('Holding CD Account')
plt.ylabel('Count')
plt.title("CD Account Distribution")


plt.subplot(6,2,5)
df['Online'].value_counts().plot(kind="bar", align='center',color = 'green',edgecolor = 'black')
plt.xlabel('Accessing Online Banking Facilities')
plt.ylabel('Count')
plt.title("Online Banking Distribution")


plt.subplot(6,2,6)
df['CreditCard'].value_counts().plot(kind="bar", align='center',color = 'lightblue',edgecolor = 'black')
plt.xlabel('Holding Credit Card')
plt.ylabel('Count')
plt.title("Credit Card Distribution")


#OBSERVATION: Most of them are not having security and CD account


# In[17]:


#Q3)Perform correlation analysis among all the variables - you can use Pairplot and Correlation coefficients of
#every attribute with every other attribute (5 points)

#Checking for correlation

df[['Personal Loan', 'Age', 'Income', 'CCAvg', 'Mortgage']].corr()


# In[73]:


#Pairplot

sns.pairplot(df.iloc[:,1:])


# In[18]:


sns.heatmap(df[['Personal Loan', 'Age', 'Income', 'CCAvg', 'Mortgage']].corr(), annot = True)


# In[19]:


df[['Personal Loan', 'Age', 'Income', 'CCAvg', 'Mortgage']].corr()['Personal Loan'][1:].plot.bar()


#OBSERVATION: Income and CCAvg have some correlation with the Personal Loan, Age has hardly any correlation.


# In[20]:


#Q4). One hot encode the Education variable (3 points)

df_dummies= pd.get_dummies(df, prefix='Edu', columns=['Education']) #This function does One-Hot-Encoding on categorical text


# In[21]:


# returns the names of all the columns as a list

df_dummies.head()


# In[22]:


df_dummies.columns


# In[24]:


#Q5). Separate the data into dependant and independent variables and create training and test sets out of them
#(X_train, y_train, X_test, y_test) (2 points)


#Dependant variable analysis

df["Personal Loan"].value_counts().to_frame()


# In[25]:


pd.value_counts(df["Personal Loan"]).plot(kind="bar")


# In[26]:


#Influence of few attributes on 'Personal Loan' - Dependant Variable
plt.figure(figsize=(15,15))

plt.subplot(3,1,1)
sns.scatterplot(df.CCAvg, df.Income, hue = df['Personal Loan'], palette= ['red','green'])

plt.subplot(3,1,2)
sns.scatterplot(df.Family, df.Income, hue = df['Personal Loan'], palette= ['violet','purple'])

plt.subplot(3,1,3)
sns.scatterplot(df.Income, df.Mortgage, hue = df['Personal Loan'], palette= ['yellow','green'])


# In[27]:


plt.figure(figsize=(15,15))

plt.subplot(3,1,1)
sns.scatterplot(df.Age, df.Experience, hue = df['Personal Loan'], palette= ['yellow','violet'])

plt.subplot(3,1,2)
sns.scatterplot(df.Education, df.Income, hue = df['Personal Loan'], palette= ['blue','yellow'])

plt.subplot(3,1,3)
sns.scatterplot(df.Education, df.Mortgage, hue = df['Personal Loan'], palette= ['red','yellow'])


# In[28]:


plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
sns.countplot(x="Securities Account", data=df ,hue="Personal Loan")

plt.subplot(2,2,2)
sns.countplot(x='CD Account' ,data=df ,hue='Personal Loan')


# In[29]:


sns.distplot(df[df["Personal Loan"] == 0]['Income'], color = 'b')
sns.distplot(df[df["Personal Loan"] == 1]['Income'], color = 'y')


# In[30]:


#creating training and test sets

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)


# In[31]:


train_labels = train_set.pop('Personal Loan')
test_labels = test_set.pop('Personal Loan')


# In[32]:


train_set_indep = df.drop(['Experience' ,'ID'] , axis = 1).drop(labels= "Personal Loan" , axis = 1)
train_set_dep = df["Personal Loan"]
X = np.array(train_set_indep)
Y = np.array(train_set_dep)
X_Train = X[ :3500, :]
X_Test = X[3501: , :]
Y_Train = Y[:3500, ]
Y_Test = Y[3501:, ]


# In[81]:


#Q6) Use StandardScaler( ) from sklearn, to transform the training and test data into scaled values ( fit the
#StandardScaler object to the train data and transform train and test da
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(df))
StandardScaler()
print(scaler.mean_)

print(scaler.transform(df))


# In[61]:


#Q7)Write a function which takes a model, X_train, X_test, y_train and y_test as input and returns the accuracy,
#recall, precision, specificity, f1_score of the model trained on the train set and evaluated on the test set (5
#points)

confusion_matrix = confusion_matrix(Y_Test, predicted)
print(confusion_matrix)


# In[63]:


print(classification_report(Y_Test, predicted))


# In[59]:


#Q8). Employ multiple Classification models (Logistic, K-NN, Naïve Bayes etc) and use the function from step 7
#to train and get the metrics of the model (15 points)

#logistic regression


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_Train,Y_Train)


# In[40]:


predict = logmodel.predict(X_Test)
predictProb = logmodel.predict_proba(X_Test)


# In[41]:


# Confusion Matrix
cm = confusion_matrix(Y_Test, predict)

class_label = ["Positive", "Negative"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[66]:


# Classification Report
LOG_reg=classification_report(Y_Test, predict)
print(classification_report(Y_Test, predict))


# In[45]:


# KNN Model
# Creating odd list of K for KNN
myList = list(range(1,20))

# Subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))


# In[46]:


# Empty list that will hold accuracy scores
ac_scores = []

# Perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_Train, Y_Train)
    
    # Predict the response
    Y_Pred = knn.predict(X_Test)
    
    # Evaluate accuracy
    scores = accuracy_score(Y_Test, Y_Pred)
    ac_scores.append(scores)

# Changing to misclassification error
MSE = [1 - x for x in ac_scores]

# Determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)


# In[47]:


knn = KNeighborsClassifier(n_neighbors= 13 , weights = 'uniform', metric = 'euclidean')
knn.fit(X_Train, Y_Train)    
predicted = knn.predict(X_Test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_Test, predicted)
print(acc)


# In[48]:


plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')


# In[49]:


# Confusion Matrix
cm1 = confusion_matrix(Y_Test, predicted)

class_label = ["Positive", "Negative"]
df_cm1 = pd.DataFrame(cm1, index = class_label, columns = class_label)
sns.heatmap(df_cm1, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[50]:


# Classification Report
print(classification_report(Y_Test, predicted))


# In[51]:


# Naive Model
naive_model = GaussianNB()
naive_model.fit(train_set, train_labels)

prediction = naive_model.predict(test_set)
naive_model.score(test_set,test_labels)


# In[52]:


# Confusion Matrix
cm2 = confusion_matrix(test_labels, prediction)

class_label = ["Positive", "Negative"]
df_cm2 = pd.DataFrame(cm2, index = class_label, columns = class_label)
sns.heatmap(df_cm2, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[53]:


# Classififcation Report
print(classification_report(test_labels, prediction))


# In[71]:


#Q9) Create a dataframe with the columns - “Model”, “accuracy”, “recall”, “precision”, “specificity”, “f1_score”.
#Populate the dataframe accordingly (5 points)

LOG_reg=classification_report(Y_Test, predict)
print(classification_report(Y_Test, predict))

KNN=classification_report(Y_Test, predict)
print(classification_report(Y_Test, predicted))

Naive=classification_report(Y_Test, predict)
print(classification_report(test_labels, prediction))


# In[ ]:


#Q10)Give your reasoning on which is the best model in this case (5 points)

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))

# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=12345)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Logistic Regression is the best Model as its accuracy is highest and recall is also good

#In KNN the accuracy and recall is good but the confusion matrix tells it is not the accurate model for prediction

#Naive Bayes has less accuracy and recall compared to other models


# In[ ]:




