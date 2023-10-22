#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd

data=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\House prediction data set.xlsx")
df=pd.DataFrame(data)


# In[100]:


df.head()


# In[101]:


df.isnull().sum()


# In[102]:


df.info()


# In[103]:


df.shape


# In[104]:


df.columns


# In[105]:


df=df[["date", 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']]


# In[106]:


df.head()


# In[107]:


df.drop(columns=["date"],inplace=True)


# In[108]:


df.head()


# In[109]:


import seaborn as sns
sns.histplot(x="price",data=data)


# In[110]:


from matplotlib import pyplot as plt
plt.scatter(x="bedrooms",y="bathrooms",color="red",data=data)


# In[111]:


df.head()


# In[112]:


features=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","sqft_above","sqft_basement","yr_built","yr_renovated"]

x=df[features]
y=df["price"]


# In[113]:


print(x.shape)
print(y.shape)


# In[114]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[115]:


print(x_train.shape)
print(y_train.shape)


# In[116]:


print(x_test.shape)
print(y_test.shape)


# In[117]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)


# In[118]:


y_pred=regression.predict(x_test)
print("y_pred:",y_pred)


# In[119]:


y_test


# In[123]:


from sklearn.metrics import r2_score

r2 =r2_score(y_test,y_pred)
print("r2_score:",r2 )


# In[ ]:





# In[ ]:





# In[97]:


import pandas as pd

data=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\House prediction data set.xlsx")
df=pd.DataFrame(data)

 

df=df[["date", 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']]

df.drop(columns=["date"],inplace=True)

 
features=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","sqft_above","sqft_basement","yr_built","yr_renovated"]

x=df[features]
y=df["price"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

y_pred=regression.predict(x_test)
print("y_pred:",y_pred)

from sklearn.metrics import r2_score

r2_score=r2_score(y_test,y_pred)
print("r2_score:",r2_score)


# In[ ]:




