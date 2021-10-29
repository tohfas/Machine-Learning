#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np

#Date stuff
from datetime import datetime
from datetime import timedelta

#Library for Nice graphing
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sn
get_ipython().run_line_magic('matplotlib', 'inline')

#Library for statistics operation
import scipy.stats as stats

# Date Time library
from datetime import datetime

#Machine learning Library
import statsmodels.api as sm
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[24]:


#1. Preprocessing the data (15 points)
# a. Check a few observations and get familiar with the data. (1 points)

df = pd.read_csv("C:/Users/user/Downloads/data_credit_card.csv")


# In[25]:


df.head(3)


# In[26]:


#b. Check the size and info of the data set. (2 points)
df.info()


# In[12]:


df.shape


# In[15]:


#c. Check for missing values. Impute the missing values if there is any. (2 points)
df.isnull().sum().values.sum()


# In[19]:


#Since there are missing values in the data so we are imputing them with median
df.isnull().any()


# In[20]:


# CREDIT_LIMIT  and MINIMUM_PAYMENTS has missing values so we need to remove with median.

df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(),inplace=True)

df['CREDIT_LIMIT'].count()


df['MINIMUM_PAYMENTS'].median()
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(),inplace=True)


# In[21]:


# Now again check the missing values.

df.isnull().any()


# In[22]:


df.isnull().sum().values.sum()


# In[27]:


#d. Drop unnecessary columns. (2 points)
df = df.drop(columns=['CUST_ID','TENURE'])
df.sample(7)


# In[29]:


df.shape


# In[105]:


df_copy = df.copy()


# In[106]:


#e. Check correlation among features and comment your findings. (3 points)

# Heatmap of the Pearson correlation between each pair of features
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap='viridis', vmax=0.70) # I set vmax=0.7 to easily spot correlations greater than or close to 0.7
plt.show()


# In[107]:


df=df.corr()
df


# In[108]:


# Function to get unique correlations from the correlation matrix
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# Function to get top n absolute correlation values and corresponding columns
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# In[109]:


print("Top Absolute Correlations \n")
print(get_top_abs_correlations(df, 5))


# In[110]:


# Visualization of the top correlated variables
x = get_top_abs_correlations(df, 5)
x = x.reset_index()
x.columns = ['Column_1', 'Column_2', 'Correlation' ]
x['Column Pairs'] = x['Column_1'] + '  &  ' + x['Column_2']
sns.barplot(x='Correlation', y='Column Pairs', data=x, palette='hot')
plt.title('Top Correlations')
plt.show()


#Very high correlation between PURCHASES and ONEOFF_PURCHASES


# In[111]:


# Jointplot to visualize the high correlation between PURCHASES and ONEOFF_PURCHASES
sns.jointplot(x='PURCHASES', y='ONEOFF_PURCHASES', data=df, color='lightgreen')
plt.show()


# In[101]:


df[['PURCHASES', 'ONEOFF_PURCHASES']].head(5)


# In[102]:


df[['PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']].head(5)


# In[103]:


df.drop('PURCHASES', axis=1, inplace=True)


# In[104]:


# KDE plot to visualize high correlation between PURCHASES_FREQUENCY and PURCHASES_INSTALLMENTS_FREQUENCY
sns.kdeplot( df['PURCHASES_FREQUENCY'], df['PURCHASES_INSTALLMENTS_FREQUENCY'],
                 cmap="plasma", shade=True, shade_lowest=False)
plt.show()


# In[51]:


# Dropping PURCHASES_INSTALLMENTS_FREQUENCY
df.drop('PURCHASES_INSTALLMENTS_FREQUENCY', axis=1, inplace=True)


# In[52]:



# Plot to visualize linear relationship between CASH_ADVANCE_TRX and CASH_ADVANCE_FREQUENCY
sns.lmplot(x='CASH_ADVANCE_TRX', y='CASH_ADVANCE_FREQUENCY', data=df)


# In[67]:



# Plotting Boxplots of all our features to get idea of distribution and outliers
plt.figure(figsize=(18,6))
sns.boxplot(x="value", y="variable", data=pd.melt(df))
plt.title('Boxplots of all variables', size=15)
plt.show()


# In[54]:


#f. Check distribution of features and comment your findings. (3 points)
df.describe()


# In[68]:


#g. Standardize the data using appropriate methods. (2 points)
from sklearn.preprocessing import StandardScaler


# In[69]:


# Scaling the data
# Scaling the data
scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.fit_transform(df), columns= df.columns, index=df.index)
df.head()


# In[70]:



# Replacing extreme outliers with 99th and 1st percentiles for each variable
df_kmeans = df.copy()
for i in df_kmeans.columns:
    ulimit = np.percentile(df_kmeans[i].values, 99)
    llimit = np.percentile(df_kmeans[i].values, 1)
    df_kmeans[i].loc[df_kmeans[i]>ulimit] = ulimit
    df_kmeans[i].loc[df_kmeans[i]<llimit] = llimit


# In[71]:


# Comparison of Boxplots before and after treating outliers
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,12) )
sns.boxplot(x="value", y="variable", data=pd.melt(df), ax=ax[0], palette='cool')
ax[0].title.set_text('Boxplots of original data')

sns.boxplot(x="value", y="variable", data=pd.melt(df_kmeans), ax=ax[1], palette='cool')
ax[1].title.set_text('Boxplots after treating extreme outliers')
plt.show()


# In[91]:


#3. Apply PCA to the dataset and perform all steps from Q2 on the new features generated using PCA. (15
#points)
from sklearn.decomposition import PCA


# In[92]:


scaler = StandardScaler()
scaler.fit(df_copy)
scaled_df = pd.DataFrame(scaler.fit_transform(df_copy), columns= df_copy.columns, index=df_copy.index)
scaled_df.head()


# In[84]:


#2. Build a k-means algorithm for clustering credit card data. Kindly follow the below steps and answer the
#following. (10 points)
#a. Build k means model on various k values and plot the inertia against various k values.

# Imports for kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# In[ ]:


# k-means with 10 different centroid seeds
# (init= 'k-means++' : selects initial cluster centers in a smart way to speed up convergence)


# In[85]:


# and Silhouette scores for different values of k (number of clusters).

def kmeans_analysis(df_kmeans, random_state=101):
    
    range_n_clusters = list(range(2,11))
    silhouette_scores = []
    wss = []
    
    # Taking 2 Principal Components for the dataset for purpose of visualization
    pca = PCA().fit(df_kmeans)
    X = pca.fit_transform(df_kmeans)
    
    # Looping through the values of k
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state= random_state)
        cluster_labels = clusterer.fit_predict(df_kmeans)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(df_kmeans, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", round(silhouette_avg,2))
        
        # Appending silhouette score and within cluster squared error to seperate lists for the particular k value
        silhouette_scores.append(silhouette_avg)
        wss.append(clusterer.inertia_)
        
        # Silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df_kmeans, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values =                 sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("Visualization of the clustered data on 1st and 2nd PC")
        ax2.set_xlabel("Feature space for PC_1")
        ax2.set_ylabel("Feature space for PC_2")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()
    return silhouette_scores, wss


# In[86]:


# Function to plot WSS and Silhouette scores vs No. of clusters
def make_plots(wss, silhouette_scores):
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
    
    ax1.plot(range(2, 11), wss, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=8)
    ax1.set_title('Elbow method- WSS vs k', size=13)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Within Cluster Sum of Squares (WSS)') #within cluster sum of squares
    
    ax2.plot(range(2, 11), silhouette_scores, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=8)
    ax2.set_title('Silhouette Scores vs k', size=13)
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Coefficient')
    
    plt.suptitle(('Choosing k value for kmeans'),
             fontsize=14, fontweight='bold')
    plt.show()


# In[87]:


#b. Evaluate the model using Silhouette coefficient
silhouette_scores, wss = kmeans_analysis(df_kmeans)


# In[90]:


#c. Plot an elbow plot to find the optimal value of k
make_plots(wss, silhouette_scores)


# In[89]:


#d. Which k value gives the best result?


# In[ ]:


#3. Apply PCA to the dataset and perform all steps from Q2 on the new features generated using PCA. (15
#points)
from sklearn.decomposition import PCA


# In[ ]:


cr_dummy.shape


# In[ ]:



#We have 17 features so our n_component will be 17.
pc=PCA(n_components=17)
cr_pca=pc.fit(cr_scaled)


# In[ ]:


#Lets check if we will take 17 component then how much varience it explain. Ideally it should be 1 i.e 100%
sum(cr_pca.explained_variance_ratio_)


# In[ ]:



var_ratio={}
for n in range(2,18):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)


# In[ ]:



var_ratio


# In[ ]:


#Since 6 components are explaining about 90% variance so we select 5 components

pc=PCA(n_components=6)


# In[ ]:


p=pc.fit(cr_scaled)


# In[ ]:


cr_scaled.shape


# In[ ]:


p.explained_variance_


# In[ ]:


np.sum(p.explained_variance_)


# In[ ]:


np.sum(p.explained_variance_)


# In[ ]:


var_ratio


# In[ ]:


pd.Series(var_ratio).plot()


# In[ ]:


#Since 5 components are explaining about 87% variance so we select 5 components
cr_scaled.shape


# In[ ]:


pc_final=PCA(n_components=6).fit(cr_scaled)

reduced_cr=pc_final.fit_transform(cr_scaled)


# In[ ]:



dd=pd.DataFrame(reduced_cr)


# In[ ]:


dd.head()


# In[ ]:


#So initially we had 17 variables now its 5 so our variable go reduced
dd.shape


# In[ ]:


col_list=cr_dummy.columns


# In[ ]:


col_list


# In[ ]:


pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(6)],index=col_list)


# In[ ]:


#So above data gave us eigen vector for each component we had all eigen vector value very small we can remove those variable bur in our case its not

# Factor Analysis : variance explained by each component- 
pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(6)])


# In[ ]:


#4. Create a new column as a cluster label in the original data frame and perform cluster analysis. Check the
#correlation of cluster labels with various features and mention your inferences. (Hint - Does cluster 1
#have a high credit limit?) (5 points)


#Based on the intuition on type of purchases made by customers and their distinctive behavior exhibited based on the purchase_type (as visualized above in Insights from KPI) , I am starting with 4 clusters.
from sklearn.cluster import KMeans


# In[ ]:


km_4=KMeans(n_clusters=4,random_state=123)


# In[ ]:


km_4.fit(reduced_cr)


# In[ ]:


km_4.labels_


# In[ ]:


pd.Series(km_4.labels_).value_counts()
#Here we donot have known k value so we will find the K. To do that we need to take a cluster range between 1 and 21.


# In[ ]:


#Identify cluster Error


cluster_range = range( 1, 21 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( reduced_cr )
    cluster_errors.append( clusters.inertia_ )# clusters.inertia_ is basically cluster error here.


# In[ ]:



clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:21]


# In[ ]:


# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[ ]:





# In[ ]:


#5. Comment your findings and inferences and compare the performance. Does applying PCA give a better
#result in comparison to earlier? (5 points)

from sklearn.decomposition import PCA


# In[ ]:


cr_dummy.shape


# In[ ]:


#We have 17 features so our n_component will be 17.
pc=PCA(n_components=17)
cr_pca=pc.fit(cr_scaled)


# In[ ]:


#Lets check if we will take 17 component then how much varience it explain. Ideally it should be 1 i.e 100%
sum(cr_pca.explained_variance_ratio_)


# In[ ]:


var_ratio={}
for n in range(2,18):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)


# In[ ]:


var_ratio


# In[ ]:


pc=PCA(n_components=6)


# In[ ]:




