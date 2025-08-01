# %% [markdown]
# # Clustering Agorithms

# %% [markdown]
# ## **Mall Customer Segmentation Model**

# %% [markdown]
# ## Project Scope:
# 
# Malls are often indulged in the race to increase their customers and making sales. To achieve this task machine learning is being applied by many malls already.
# 
# It is amazing to realize the fact that how machine learning can aid in such ambitions. The shopping malls make use of their customers’ data and develop ML models to target the right audience for right product marketing.
# 
# 
# **Your role:** Mall Customer data is an interesting dataset that has hypothetical customer data. It puts you in the shoes of the owner of a supermarket. You have customer data, and on this basis of the data, you have to divide the customers into various groups.
# 
# **Goal:** Build an unsupervised clustering model to segment customers into correct groups.
# 
# **Specifics:**
# 
# * Machine Learning task: Clustering model
# * Target variable: N/A
# * Input variables: Refer to data dictionary below
# * Success Criteria: Cannot be validated beforehand
# 

# %% [markdown]
# ## Data Dictionary:
# 
# * **CustomerID:** Unique ID assigned to the customer
# * **Gender:** Gender of the customer
# * **Age:** Age of the customer
# * **Income:** Annual Income of the customers in 1000 dollars
# * **Spending_Score:** Score assigned between 1-100 by the mall based on customer' spending behavior

# %% [markdown]
# ## **Data Analysis and Data Prep**

# %% [markdown]
# ### Loading all the necessary packages

# %%
import streamlit as st

st.title("Clustering_Solution")

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ### Reading the data

# %%
df = pd.read_csv(r'C:\Users\mohsa\Downloads\mall_customers.csv')
df.head()

# %%
df.shape

# %%
# Check some quick stats of the data
df.describe()

# %%
df.corr(numeric_only=True)

# %%
df.corr(numeric_only=True, method='spearman' )

# %%
# let's plot a pairplot
sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
plt.show()

# %% [markdown]
# * As a mall owner you are interested in the customer spending score. If you look at spending vs Age, you can observe that the spending score is high for customers between age 20-40, and relatively low for customers beyond 40.
# <br><br>
# * Remember, K-means clustering is sensitive to outliers. So, if you see any guilty outliers you should consider removing them.

# %%
# import kmeans model
from sklearn.cluster import KMeans

# %% [markdown]
# #### We will build a model with only 2 features for now to visualise it, and later we will add more feature' and use the evaluation metric silhouette measure.

# %%
# Let' train our model on spending_score and annual_income
kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])

# %%
# check your cluster centers
kmodel.cluster_centers_

# %%
# Check the cluster labels
kmodel.labels_

# %%
# Put this data back in to the main dataframe corresponding to each observation
df['Cluster'] = kmodel.labels_

# %%
# check the dataset
df.head()

# %%
# check how many observations belong to each cluster
df['Cluster'].value_counts()

# %%
# Let' visualize these clusters
sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
plt.show()

# %% [markdown]
# #### Visually we are able to see 5 clear clusters. Let's verify them using the Elbow and Silhouette Method

# %% [markdown]
#     

# %% [markdown]
# ## 1. Elbow Method

# %% [markdown]
# * We will analyze clusters from 3 to 8 and calculate the WCSS scores. The WCSS scores can be used to plot the Elbow Plot.
# 
# * WCSS = Within Cluster Sum of Squares

# %%
# try using a for loop
k = range(3,9)
K = []
WCSS = []
for i in k:
    kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    K.append(i)

# %%
K, WCSS

# %%
# Store the number of clusters and their respective WSS scores in a dataframe
wss = pd.DataFrame({'cluster': K, 'WCSS_Score':WCSS})
wss

# %%
# Now, plot a Elbow plot
wss.plot(x='cluster', y = 'WCSS_Score')
plt.xlabel('No. of clusters')
plt.ylabel('WSS Score')
plt.title('Elbow Plot')
plt.show()

# %% [markdown]
# We get 5 clusters as a best value of k using the WSS method.

# %% [markdown]
#     

# %% [markdown]
# ## 2. Silhouette Measure
# 
# Check the value of K using the Silhouette Measure

# %%
# import silhouette_score
from sklearn.metrics import silhouette_score

# %%
# same as above, calculate sihouetter score for each cluster using a for loop

# try using a for loop
k = range(3,9) # to loop from 3 to 8
K = []         # to store the values of k
ss = []        # to store respective silhouetter scores
for i in k:
    kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
    ypred = kmodel.labels_
    sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
    K.append(i)
    ss.append(sil_score)

# %%
ss

# %%
# Store the number of clusters and their respective silhouette scores in a dataframe
wss['Silhouette_Score']=ss

# %%
wss

# %% [markdown]
# ### Silhouette score is between -1 to +1
# 
# closer to +1 means the clusters are better

# %%
# Now, plot the silhouette plot
wss.plot(x='cluster', y='Silhouette_Score')
plt.xlabel('No. of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Plot')
plt.show()

# %% [markdown]
#     

# %% [markdown]
# #### Conclusion: Both Elbow and Silhouette methods gave the optimal value of k=5

# %% [markdown]
#     

# %% [markdown]
# ## Now use all the available features and use the k-means model.
# 
# * Remember, now you cannot visualise the clusters with more than 2 features.
# * So, the optimal number of clusters can be only determined by Elbow and Silhouette methods.

# %%
# Train a model on 'Age','Annual_Income','Spending_Score' features
k = range(3,9)
K = []
ss = []
for i in k:
    kmodel = KMeans(n_clusters=i).fit(df[['Age','Annual_Income','Spending_Score']], )
    ypred = kmodel.labels_
    sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
    K.append(i)
    ss.append(sil_score)

# %%
# Store the number of clusters and their respective silhouette scores in a dataframe
Variables3 = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
Variables3

# %%
# Now, plot the silhouette plot
Variables3.plot(x='cluster', y='Silhouette_Score')

# %% [markdown]
#     

# %%
# try using a for loop
k = range(3,9)
K = []
WCSS = []
for i in k:
    kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score','Age']])
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    K.append(i)

# %%
# Store the number of clusters and their respective WSS scores in a dataframe
wss = pd.DataFrame({'cluster': K, 'WCSS_Score':WCSS})
wss

# %%
# Now, plot a Elbow plot
wss.plot(x='cluster', y = 'WCSS_Score')
plt.xlabel('No. of clusters')
plt.ylabel('WSS Score')
plt.title('Elbow Plot')
plt.show()

# %% [markdown]
# #### Conclusion: With 3 features we now have the optimal value of k=6

# %% [markdown]
# ### Exercise:
# 
# Use argument `init=kmeans++` as a hyperparameter while training the model.
# 
# KMEANS++ internally analyzes the patterns of the data. Such as the shape of data (whether it is spherical, rectangle, oval etc.) and then initializes the centroids. Thus, assigning the clusters in a smart way.


