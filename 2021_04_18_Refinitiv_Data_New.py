#!/usr/bin/env python
# coding: utf-8

# In[161]:


import eikon as ek
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#Eikon Key eingeben String!
ek.set_app_key('45e1a54a571d4128b534f34665d6a6e056b4421c')

#creds = pd.read_csv('refinitiv_api.csv', sep = ',', encoding='latin-1')

#ek.set_app_key(creds.iloc[1,1])


# In[162]:


ticker = ['BTC=','LTC=','ETH=','BCH=','NEO=','DASH=','XRP='] #wäre gut, wenn man sie nicht manuell eingeben muss, sondern aus einer Liste ziehen.
print('Retrieving Data for',ticker)

df = ek.get_timeseries(ticker, interval = 'daily', fields = 'Close', start_date = '2010-01-01')

df.head()


# In[163]:


# Gleichzeichen der Tickers entfernen. Dies v.a. damit es keine Konflikte gibt. Evtl. darf man ein Excel, Bild, ... nicht mit einem "=" im Namen abspeichern.

df.columns = df.columns.str.replace('=', '')

df.head()


# In[164]:


df.describe()


# In[165]:


# scheint keine Nullwerte zu geben

df.isnull().sum()


# In[166]:


df.corr()


# In[167]:


# Heatmap wird erstellt

def visualize_data(df):
    df_corr = df.corr()
        
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    heatmap = ax.pcolor(data,cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor = False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
    


# In[168]:


visualize_data(df)


# In[18]:


# irgendwie das dataframe in einheitlich grosse Stücke unterteilen, damit man die Stücke analysieren kann.

# Funktioniert noch nich wie gewünscht.

# ideal wäre, wenn man automatisch sub-dataframes erstellen kann für jeden Ticker

np.array_split(df['BTC='], 3)


# In[183]:


#daily returns berechnen alle auf einmal

# Achtung, diese Funktion funktioniert endlos!
# Bitte nur einmal laufen lassen, da sonst daily returns von daily returns berechnet werden

# Aktuell liegt noch ein Problem bei: ich möchten len(ticker), damit ich weiss wie viele Spalten
# ich beachten soll. So wie wir es bei Bachmann hatten, dass man ein i = 0 setzt und am Ende i =+ 1
# aber das funktioniert irgenwie nicht... im schlimmsten Fall lassen wir es so :-)

def daily_returns():
    for col in df:
        dailyret_col = 'dr_' + col
        df[dailyret_col] = df[col].pct_change(1)
        print(col+' done')
        
        
# weiteres Problem: ich möchte hier ein neues Dataframe erstellen, welches nur die daily returns enthält
# so kann man die Headmap individuell erstellen lassen und kann immer noch beide Dataframes zusammenlegen


# In[184]:


daily_returns()


# In[185]:


df.head()


# In[113]:


#Plots erstellen

# Achtung! Unbedingt einen Ordner "Charts" erstellen, wo dieses Skript gespeichert ist.

def plots_save():
    for col in df.iteritems():
        print('Plotting '+col[0])
        fig, ax = plt.subplots()
        indicator= df[col[0]]
        ax.plot(indicator, alpha=0.9, color='blue')
        plt.title(col[0])
        plt.savefig('Charts\Plot_'+col[0]+'.png')
        plt.close()


# In[114]:


plots_save()


# In[117]:


# File abspeichern

df.to_csv('Crypto_group_F.csv')


# In[119]:


df.corr()


# In[ ]:


for col in df.columns[1:]:
    # create a subdf with desired columns:
    subdf = df[['GeneID',col]]
    # append subdf to list of df: 
    outdflist.append(subdf)


# In[186]:


df_dr = df[['dr_BTC','dr_LTC','dr_ETH','dr_BCH','dr_NEO','dr_DASH','dr_XRP']]
df_ticker = df[['BTC','LTC','ETH','BCH','NEO','DASH','XRP']]


# In[187]:


visualize_data(df_dr)


# In[189]:


visualize_data(df_ticker)


# In[190]:


visualize_data(df)


# In[111]:


ticker = ['BTC=']
print('Retrieving Data for',ticker)

df = ek.get_timeseries(ticker, interval = 'daily', fields = 'Close', start_date = '2010-01-01')

print('\nCheck, if Data was pulled:\n', df)
#df.head()

#Creating first Plot
print('\nPlotting the values to get a first impression:')
#time.sleep(5)
#for i in ticker:
    
plt.plot(df)
plt.ylabel('USD')
plt.xlabel('Year')
plt.show()

#Anzahl Werte die fehlen
print('\nCounting missing values:', df.iloc[0].isnull().sum())
if df.iloc[0].isnull().sum() == 0:
    print('\nThere are no missing values!')
else:
    print('\nThere are', df.iloc[0].isnull().sum(), 'missing values!')


# In[57]:


df.describe()


# # Exploratory Data Analysis (EDA)

# ## Test for normality

# In[113]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import numpy as np
from scipy import stats


df = stats.norm.rvs(loc = 2.5, scale = 2, size = 100)
shapiro_test = stats.shapiro(df)
print(shapiro_test)


# In[ ]:





# ## Test for Outliers

# In[118]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import scikit_posthocs

x = np.array([-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06, 0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01])
scikit_posthocs.outliers_tietjen(df, 2) #ursprünglich war in der Klammer (x,2)


# In[ ]:





# In[ ]:





# ## Test for correlation

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy.stats import pearsonr

data1 = stats.norm.rvs(loc=3, scale=1.5, size=20)
data2 = stats.norm.rvs(loc=-5, scale=0.5, size=20)
stat, p = pearsonr(data1, data2)
print(stat, p)


# In[ ]:





# ## Test of homogeneity

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import scipy
import scipy.stats
from scipy.stats import chisquare

data1 = stats.norm.rvs(loc=3, scale=1.5, size=20)
data2 = stats.norm.rvs(loc=-5, scale=0.5, size=20)
chisquare(data1, data2)


# In[ ]:





# ## Test for equality of distribution

# In[120]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy import stats

x = np.linspace(-25, 17, 6)
stats.kstest(df, 'norm') #ursprünglich war in der Klammer (x,'norm')


# In[ ]:





# # Regression and Classification

# ## Test for heteroscedasticity

# In[ ]:





# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip

expr = ‘y_var ~ x_var’
y, X = dmatrices(expr, df, return_type=’dataframe’)
keys = [‘LM stat’, ‘LM test p-value’, ‘F-stat’, ‘F-test p-value’]
results = het_white(olsr_results.resid, X)
lzip(keys, results)


# In[ ]:





# ## Test or multicollinearity

# In[ ]:





# In[ ]:





# ## Test of the significance of regression coefficients

# In[ ]:





# In[ ]:





# ## ANOVA for regression or classification model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




