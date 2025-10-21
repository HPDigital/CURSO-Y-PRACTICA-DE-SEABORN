"""
SEABORN
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


# In[2]:


df=pd.read_excel(f'C:/Users/HP/Desktop/TESIS/EMPRESA_NDA/DATOS/DATOS_NDA.xlsx', parse_dates=True)


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df['Dia']=df['Fecha'].dt.weekday
dia = {0:'Lunes', 1:'Martes', 2:'Miercoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}

def nombredia(a):
    if a in dia:
        return(dia[a])

df['Dia']= df['Dia'].apply(nombredia)


# In[6]:


df.head()


# In[7]:


sns.relplot(x='MontoFinalProd', y= 'Cantidad', data=df, hue='Departamento')


# In[8]:


sns.relplot(x='MontoFinalProd', y= 'Cantidad', data=df, hue='Dia')


# In[9]:


filt= (df['Cantidad']>500) | (df['MontoFinalProd']>10000)
df1=df[filt]
sns.relplot(x='MontoFinalProd', y= 'Cantidad', data=df1, hue='Dia', style='Grandes_Categorias', col='Departamento', col_wrap=3)


# In[10]:


df1.shape


# In[11]:


filt2= (df['Cantidad']<500) & (df['MontoFinalProd']<10000) & (df['Cantidad']>0) & (df['MontoFinalProd']>0)
df2=df[filt2]
sns.relplot(x='MontoFinalProd', y= 'Cantidad', data=df2, hue='Dia', style='Grandes_Categorias', col='Departamento', col_wrap=3)


# In[12]:


df2.info()


# In[14]:


#def division(x,y):
#   return(x/y)
#df2['Precio']= df2.apply(division(df2['MontoFinalProd'],df2['Cantidad']))


# In[ ]:


#df2.info()


# In[ ]:





# In[15]:


sns.displot(df2, x='Departamento', bins=1, aspect=2, hue='Grandes_Categorias')


# In[16]:


sns.displot(df2, x='Departamento', bins=1, aspect=2, hue='Grandes_Categorias', multiple='stack')


# In[17]:


sns.displot(df2, x='Departamento', bins=1, aspect=2, hue='Grandes_Categorias', multiple='dodge', col='Departamento', col_wrap=2)


# In[18]:


sns.displot(df2, x="Dia", aspect=2)


# In[19]:


sns.displot(df2, x="Departamento", aspect=2)


# In[20]:


df2['Producto'].value_counts(normalize=True)


# In[21]:


sns.displot(df2['Producto'].value_counts(), kind="kde")


# In[22]:


prod_filt=(df2['Producto']=='desinfectante ola clorito 1l') & (df2['Cantidad']<11)
sns.displot(data=df2[prod_filt], x='Cantidad', kind="kde")


# In[23]:


prod_filt=(df2['Producto']=='desinfectante ola clorito 1l') & (df2['Cantidad']<11)
sns.displot(data=df2[prod_filt], x='Cantidad')


# In[25]:


#sns.displot(data=df2, x='Departamento', y=, aspect=2)


# In[26]:


sns.displot(data=df2, x='Departamento', y='Dia', aspect=2)


# In[27]:


sns.displot(data=df2, x='Departamento', y='MontoFinalProd',aspect=2)


# In[28]:


sns.relplot(data=df2, x='MontoFinalProd', y='Cantidad', hue='Dia')
sns.rugplot(data=df2, x='MontoFinalProd', y='Cantidad')


# In[29]:


#sns.pairplot(df2)


# In[30]:


sns.catplot(data= df2, x='Dia', y='MontoFinalProd')


# In[31]:


#sns.catplot(data= df2, x='Dia', y='MontoFinalProd', kind='swarm')


# In[32]:


sns.catplot(data= df2, x='Dia', y='MontoFinalProd', kind='box')


# In[ ]:


#df2.boxplot(['Precio'])


# In[ ]:


#filt_precio=df2['Precio']>200
#df2[filt_precio]


# In[33]:


#filt_producto= df2['Producto']=='liz sanitizador neutro 2x5l - 2 und x caja'
#df2[filt_producto]['Precio'].describe()


# In[34]:


sns.catplot(x='Grandes_Categorias', y='MontoFinalProd', data=df2, kind='bar')


# In[35]:


sns.catplot(data=df2, x='Grandes_Categorias', y='MontoFinalProd', kind='box')


# In[36]:


sns.catplot(data=df2, x='Grandes_Categorias', y='MontoFinalProd',row='Departamento', col='Dia', kind='box')


# In[37]:


sns.catplot(x='Grandes_Categorias', y='MontoFinalProd',row='Departamento', col='Dia' ,data=df2, kind='bar')


# In[38]:


sns.catplot(data=df2, x='Grandes_Categorias', y='MontoFinalProd', kind='violin')


# In[39]:


df2.columns


# In[40]:


df3= pd.get_dummies(df2, columns=['Producto','Departamento','Grandes_Categorias', 'Dia'], drop_first=True)
df3.head()


# In[42]:


col_list= df3.columns.to_list()


# In[47]:


df=df3.drop(['PedidoId','MontoProd','Fecha','Id_Cliente','CodigoProducto'], axis=1)


# In[50]:


col_features= df.columns.to_list()


# In[51]:


import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[col_features])


kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[ ]:





# In[52]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[col_features])

kmeans = KMeans(init="random", n_clusters=7, n_init=10, max_iter=300, random_state=42)

kmeans.fit(scaled_features)

df['Cluster'] = kmeans.labels_


print(kmeans.inertia_)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)


# In[53]:


df.head()


# In[57]:


df2['Cluster']=df['Cluster']


# In[58]:


df2.head()


# In[61]:


sns.set_style("white")
sns.relplot(x='Cantidad', y='MontoFinalProd',hue='Cluster', data=df2)


# In[62]:


sns.set_style("white")
sns.relplot(x='Cantidad', y='MontoFinalProd',hue='Cluster', row='Departamento', col='Grandes_Categorias',data=df2)


# In[63]:


sns.set_style("white")
sns.catplot(x='Cluster', y='MontoFinalProd', row='Departamento', col='Grandes_Categorias',kind='box',data=df2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






if __name__ == "__main__":
    pass
