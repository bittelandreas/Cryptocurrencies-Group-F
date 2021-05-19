#!/usr/bin/env python
# coding: utf-8

# In[1]:


import eikon as ek
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import sqlite3
from sqlite3 import Error
from scipy import stats
import scipy as sp
import statsmodels.api as sm
import pylab
import statistics
from statsmodels.tsa.stattools import adfuller
from openpyxl import load_workbook
from openpyxl import Workbook
from datetime import date
from OLS import OLS

ek.set_app_key('ed89d084197d408686e3cfa56e3356daad46c055')

print('Done')


# # Task 4

# In[2]:


ticker = ['BTC=','LTC=','ETH=', 'BCH=', 'XRP=','EUR=','CHF=','GBP=','JPY=','CNY=','CAD=', 'AUD=','MXN=','CHFEUR=R', 'GBPEUR=R', 'JPYEUR=R','CNYEUR=R','CADEUR=R','AUDEUR=R','MXNEUR=R'] #wäre gut, wenn man sie nicht manuell eingeben muss, sondern aus einer Liste ziehen.
print('Retrieving Data for',ticker)

df1 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2019-01-01', end_date = '2019-04-11')
df2 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2019-04-12', end_date = '2019-09-08')
df3 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2019-09-09', end_date = '2020-02-05')
df4 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2020-02-06', end_date = '2020-07-04')
df5 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2020-07-05', end_date = '2020-12-01')
df6 = ek.get_timeseries(ticker, fields = 'CLOSE',interval = 'daily', calendar = 'tradingdays', start_date = '2020-12-02' , end_date = '2021-04-30')

df = pd.concat([df1,df2,df3,df4,df5,df6], axis = 0)

df.columns = df.columns.str.replace('=', '')

#Save data to an excel sheet
df.to_excel('AQM_Data.xlsx')
print('Retrieving Data successful.')


# In[3]:


df_p = pd.read_excel('AQM_Data.xlsx')
df_p = df_p.set_index(df_p['Date'])
df_p = df_p.drop(axis=1, columns = 'Date')


# Cleaning Python Data

# In[9]:


df_clean=pd.DataFrame(df_p)
df_clean.dropna(axis='rows',inplace=True)

print('table size before cleaning:', df.shape)
print('table size after cleaning:', df_clean.shape)

df_p = df_clean
df = df_p


# Read & clean excel Data

# In[8]:


df_e = pd.read_excel('AQM_task_3_v3.xlsx')
df_e = df_e.set_index(df_e['Date'])
df_e = df_e.drop(axis=1, columns = 'Date')


# In[6]:


df_clean2=pd.DataFrame(df_e)
df_clean2.dropna(axis='rows',inplace=True)

print('table size before cleaning:', df.shape)
print('table size after cleaning:', df_clean.shape)

df_e = df_clean2


# Check if the values are the same

# In[11]:


ticker_new = [s.replace("=", "") for s in ticker]

for i in ticker_new:
    s = df_e [i].sum() - df_p[i].sum()
    m = df_e [i].mean() - df_p[i].mean()
    min = df_e [i].min() - df_p[i].min()
    max = df_e [i].max() - df_p[i].max()
    std = df_e [i].std() - df_p[i].std()
    if (s + m + min + max + std) == 0:
        print(i,': Data are equal.')        
    else:
        print (i,': Data are NOT equal.')


# # Task 5

# Create a folder for the Charts, Histograms etc.

# In[8]:


def create_folder(directory):
    try:
        if os.path.exists(directory):
            print("Folder '%s' already exists." %directory, "\nFiles will be stored in '%s'." %directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder '%s' has been created." %directory, "\nFiles will be stored in '%s'." %directory)
    except OSError:
        print("Creation of the folder failed.")


# In[9]:


create_folder('Charts')


# In[10]:


def plots_save():
    for col in df.iteritems():
        print('Plotting '+col[0])
        fig, ax = plt.subplots()
        indicator= df[col[0]]
        ax.plot(indicator, alpha=0.9, color='blue')
        plt.title(col[0])
        plt.savefig('Charts\Plot_'+col[0]+'.png')
        plt.close()


# In[11]:


plots_save()


# Check if values are missing

# In[12]:


ticker_new = [s.replace("=", "") for s in ticker]

for i in ticker_new:
    print('')
    if (df_e[i].isnull().sum().any() == False) and (df_p[i].isnull().sum().any() == False):
        print(i,': There isn\'t any wrong or missing data.')
        print('No cleaning needed.')
    else:
        if df_e[i].isnull().sum().any() == True:          
            print(i,': There is',df_e[i].isnull().sum(),'wrong or missing data in the Excel data!')

            print('Table size before cleaning:', df_e.shape)
            df_e.dropna(axis = 'rows', how = 'any', inplace = True)

            print('Table size after cleaning:', df_e.shape)
        else:          
            print(i,': There is',df_p[i].isnull().sum(),'wrong or missing data in the Python data!')

            print('Table size before cleaning:', df_p.shape)
            df_p.dropna(axis = 'rows', how = 'any', inplace = True)

            print('Table size after cleaning:', df_p.shape)

print('\nSince the data is identical, we can work only with the data from python (df_p = df)')


# # Task 6

# Initial Analysis

# In[13]:


df.describe()


# # Task 7

# Create table in SQLite database

# In[38]:


def create_table():
    db_file = 'AQM_Group_F_Crypto.db'
    conn = sqlite3.connect(db_file) 
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS daily_data                 (Date TIMESTAMP, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')
    print('Table successfully created, if it didn\'t exist.')
    
    conn.commit()
    c.close()
    conn.close()


# In[39]:


create_table()


# In[40]:


def data_entry():
    db_file = 'AQM_Group_F_Crypto.db'
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    
    index1 = df.index.strftime('%Y-%m-%d')
    df['Date'] = index1
    
    records_to_insert = df.values.tolist()
    
    c.executemany("INSERT INTO daily_data (BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR, Date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records_to_insert)

    conn.commit()
    c.close()
    conn.close()
    
    print('Data insertion successful.')


# In[41]:


data_entry()


# # Task 8

# Create an excel-sheet to save the results

# In[13]:


wb = Workbook()
ws = wb.active
wb.save(filename = 'AQM_Solutions.xlsx')
writer = pd.ExcelWriter('AQM_Solutions.xlsx', engine = 'openpyxl')


# Plot Histograms

# In[43]:


for col in df.iteritems(): 
    print('Histogram ' + col[0])
    fig, ax = plt.subplots()
    indicator = df[col[0]]
    ax.hist(indicator, alpha=0.9, color='blue',bins=20)
    plt.title(col[0]+' Histogram')
    plt.savefig('Charts\Hist_'+col[0]+'.png')
    plt.close()
print('\nHistograms successful.')


# ## Shapiro-Wilk normality test
#     
#     Assets are clearly not normally distributed

# In[44]:


ticker_new = [s.replace("=", "") for s in ticker]
alpha = 0.05

def nans(shape, dtype = float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

column_names = ['Statistics'] + ticker_new
table_width = len(ticker_new)+1
table_empty = nans([1, table_width])
table_1 = pd.DataFrame(table_empty, columns = column_names)
table_1.iloc[0, 0] = 'P-Value'

s = 1

for i in ticker_new:
    print('\n'+i+':')
    
    #Q-Q Plot
    mean1 = statistics.mean(df[i]) 
    st_dev1 = statistics.stdev(df[i])
    obs_count1 = len(df[i])
    sm.qqplot(df[i], loc = mean1, scale = st_dev1, line='s')
    
    plt.title(i+' Q-Q Plot')
    plt.savefig('Charts\QQ_'+i+'.png') 
    plt.close()
    print('Q-Q Plot saved.')
    
    time.sleep(0.05)
    
    #boxplot
    boxplot1 = df.boxplot(column=[i])
    
    plt.title(i+' Boxplot')
    plt.savefig('Charts\Box_'+i+'.png') 
    plt.close()
    print('Boxplot saved.')
    
    time.sleep(0.05)
    
    #normality test
    stat,p = sp.stats.shapiro(df[i])

    print('p-value for Shapiro-Wilk:',p)
    
    table_1.iloc[0, s] = p
    s = s+1
    
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
        
table_1.to_excel(writer,'Shapiro-Wilk', index=False)
writer.save()

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Shapiro_Wilk             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

table_1 = table_1.values.tolist()

c.executemany("INSERT OR REPLACE INTO Shapiro_Wilk                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_1)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')


# ## Jarque Bera
# 
#     Confirms that Data is not normally distributed

# In[45]:


ticker_new = [s.replace("=", "") for s in ticker]
alpha = 0.05



def nans(shape, dtype = float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

column_names = ['Statistics'] + ticker_new
table_width = len(ticker_new)+1
table_empty = nans([5, table_width])
table_2 = pd.DataFrame(table_empty, columns = column_names)
table_2.iloc[0, 0] = 'Mean'
table_2.iloc[1, 0] = 'Median'
table_2.iloc[2, 0] = 'Skewness'
table_2.iloc[3, 0] = 'Kurtosis'
table_2.iloc[4, 0] = 'Jarque Bera'

s = 1
for i in ticker_new:       
    print('\n'+i+':')

    #Mean
    mean2 = np.mean(df[i])
    print('Mean:', round(mean2,5))

    table_2.iloc[0, s] = mean2    
  
    #Median
    median2 = np.median(df[i])
    print('Median:', round(median2,5)) 

    table_2.iloc[1, s] = median2
    
    #Skewness
    skew2 = stats.skew(df[i])
    print('Skewness:', round(skew2,5)) 

    table_2.iloc[2, s] = skew2
    
    #Kurtosis
    kurtosis2 = stats.kurtosis(df[i])
    print('Kurtosis:', round(kurtosis2,5))
    
    table_2.iloc[3, s] = kurtosis2

    #Jarque Bera
    JB,pjb = stats.jarque_bera(df[i])
    print('Jarque Bera p-value:', pjb)

    table_2.iloc[4, s] = pjb
    
    s = s + 1
    
    if pjb > alpha:
        print('Sample is normally distributed (fail to reject H0)')
    else:
        print('Sample is not normally distributed (reject H0)')

table_2.to_excel(writer,'Mean_Med_K_Skew', index=False)
writer.save()

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Mean_Med_K_Skew             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

table_2 = table_2.values.tolist()

c.executemany("INSERT OR REPLACE INTO Mean_Med_K_Skew                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_2)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')


# ## Stationarity Test
# 
#     In both legs, Prices are non-stationary

# In[46]:


ticker_new = [s.replace("=", "") for s in ticker]

def nans(shape, dtype = float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

column_names = ['Statistics'] + ticker_new
table_width = len(ticker_new)+1
table_empty = nans([2, table_width])
table_stat = pd.DataFrame(table_empty, columns = column_names)
table_stat.iloc[0, 0] = 'Lag 1'
table_stat.iloc[1, 0] = 'Lag 2'


s = 1
for i in ticker_new:
    print('\n'+i+':')
    for lags in range(1,3):
        print('  Lag number:',lags)
                
        adf_library = adfuller(df[i], maxlag = lags, regression = 'nc', autolag = None)
        
        print('     ADF P-value for Prices (Level):',"%.5f" % adf_library[1],)
        if lags < 2:
            table_stat.iloc[0, s] = adf_library[1]
            
        else:
            table_stat.iloc[1, s] = adf_library[1]
        
        if adf_library[1] > alpha:
            print('     Sample is non-stationary (fail to reject H0)')
        else:
            print('     Sample is stationary (reject H0)')
            
    s = s + 1

table_stat.to_excel(writer,'Stationarity', index=False)
writer.save()

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Stationarity             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

table_stat = table_stat.values.tolist()

c.executemany("INSERT OR REPLACE INTO Stationarity                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_stat)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')


# ## Stationarity Absolute Returns
# 
#     In both legs, absolute Returns are stationary

# In[47]:


ticker_new = [s.replace("=", "") for s in ticker]

def nans(shape, dtype = float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

column_names = ['Statistics'] + ticker_new
table_width = len(ticker_new)+1
table_empty = nans([2, table_width])
table_stat2 = pd.DataFrame(table_empty, columns = column_names)
table_stat2.iloc[0, 0] = 'Lag 1'
table_stat2.iloc[1, 0] = 'Lag 2'

s = 1
for i in ticker_new:
    print('\n'+i+':')
    for lags in range(1,3):
        print('  Lag number:',lags)
                
        adf_library = adfuller(df[i], maxlag = lags, regression = 'nc', autolag = None)
        
        adf_library_d = adfuller(np.diff(df[i]), maxlag = lags, regression='nc', autolag = None)

        print('     ADF P-value for Absolute Returns (1st Difference):',"%.5f" % adf_library_d[1])
        
        if lags < 2:
            table_stat2.iloc[0, s] = adf_library_d[1]
            
        else:
            table_stat2.iloc[1, s] = adf_library_d[1]
        
        if adf_library_d[1] > alpha:
            print('     Sample is non-stationary (fail to reject H0)')
        else:
            print('     Sample is stationary (reject H0)')
       
    s = s + 1

table_stat2.to_excel(writer,'Stationarity_Absolute', index=False)
writer.save()

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Stationarity_Absolute             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

table_stat2 = table_stat2.values.tolist()

c.executemany("INSERT OR REPLACE INTO Stationarity_Absolute                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_stat2)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')


# ## Cointegration Test
# 
# Cointegration can be seen between the cryptocurrencies but not between vanilla currencies. Also, vanilla and cryptocurrencies seem not to be cointegrated

# In[14]:


import statsmodels.api as sm

df_clean=pd.DataFrame(df_p)
df_clean.dropna(axis='rows',inplace=True)
df = df_clean
df['Constant']= 1

x_list = ticker_new
y_list = ticker_new
max_lags = 2
counter_total = 1
counter_lib_ECMLib = 1
counter_lib_ECMLib_n_Coded = 1

def nans(shape, dtype = float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

column_names = ['Statistics'] + ticker_new
table_width = len(ticker_new)+1
table_empty = nans([4, table_width])
table_stat3 = pd.DataFrame(table_empty, columns = column_names)
table_stat3.iloc[0, 0] = 'Success'
table_stat3.iloc[1, 0] = 'Total by asset'
table_stat3.iloc[2, 0] = 'Lag 1 success'
table_stat3.iloc[3, 0] = 'Lag 2 success'
s = 1

print('Number of identified cointegrations using the ECM Library by asset: ')

for y_name in y_list: 
    
    y = df[y_name]
    counter_lib_ECMLib_success_temp = 0
    counter_total_byasset = 0
    lags_success = []
    R2_success = []
    
    for lags in range(0,max_lags,1):
    
        for i in x_list:
            
            if not y_name == i :
                counter_total = counter_total+1
                counter_total_byasset = counter_total_byasset+1
                
                x_nc = df[i]
                x = df[[i,'Constant']]
                
                if y.shape > x_nc.shape: 
                    temp = y.drop(y.index[0])
                    y = temp
                elif y.shape < x_nc.shape:
                    temp = x_nc.drop(x_nc.index[0])
                    d_i = temp
                if x_nc.shape != y.shape: print('Warning Data Problem: Y',y.shape, 'X:', x_nc.shape)

                    
                #Step 1: Stationarity Tests for the Residual between Y  and other variables
                
               
                step1_myols = OLS(y,x)
                step1_est_resid = step1_myols['est_resid']
                
                adf_library = adfuller(step1_est_resid, maxlag = lags, regression = 'nc',autolag = None)
                
                #Step 2
                
                d_y = y.diff()[1:]
                d_i = x_nc.diff()[1:]
                
                d_i_res = pd.concat([x_nc.diff()[1:], step1_est_resid.shift(1)[1:]], axis=1)
                d_i_res.columns = [i,'est_res_1']
                
                if d_y.shape[0] != d_i_res.shape[0]: print('Warning Data Problem: dY',d_y.shape[0], 'dX_rres:', d_i_res.shape[0])
                
                ECM_results_lib = sm.tsa.stattools.coint(y, df[i], trend = 'c', maxlag = lags) 
                
                if ECM_results_lib[1]<0.1:
                    counter_lib_ECMLib = counter_lib_ECMLib+1
                    
                    OLS_results_lib = sm.OLS(d_y,d_i_res,hasconst = None).fit()
                    OLS_results_code = OLS(d_y,d_i_res)
                    
                    counter_lib_ECMLib_success_temp = counter_lib_ECMLib_success_temp + 1
                    lags_success.append(lags)
                    R2_success.append(OLS_results_code['R_2'])
    
                       
    table_stat3.iloc[0, s] = counter_lib_ECMLib_success_temp
    table_stat3.iloc[1, s] = counter_total_byasset
    table_stat3.iloc[2, s] = lags_success.count(0)
    table_stat3.iloc[3, s] = lags_success.count(1)

    s = s+1

  
    print(y_name,':',counter_lib_ECMLib_success_temp,'out of total',counter_total_byasset,'¦cointegration identified at ADF lags:'
          ,lags_success)

table_stat3.to_excel(writer,'Cointegration', index=False)
writer.save()    
    
print('\n''total number of regressions conducted:',counter_total-1)
print('total number of identified cointegrations using only ECM Library:',counter_lib_ECMLib-1)
print('total number of identified cointegrations using both ECM Library and own code:',counter_lib_ECMLib_n_Coded-1)

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Cointegration             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

table_stat3 = table_stat3.values.tolist()

c.executemany("INSERT OR REPLACE INTO Cointegration                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_stat3)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')


# ## Granger Causality
# 
# Causality can be observed between cryptocurrencies and vanilla currencies for themselves. Only a between a few vanilla and cryptocurrencies a causality can be identified

# In[57]:


ticker_new = [s.replace("=", "") for s in ticker]

df_p = pd.read_excel('AQM_Data.xlsx')
df_p = df_p.set_index(df_p['Date'])
df_p = df_p.drop(axis=1, columns = 'Date')
df_clean=pd.DataFrame(df_p)
df_clean.dropna(axis='rows',inplace=True)
df = df_clean
data = pd.DataFrame(df)
laenge = list(range(1,len(ticker)+1))
df.columns = laenge

dataset = pd.DataFrame(np.zeros((len(laenge), len(laenge))), columns=laenge, index=laenge)

test = 'ssr-chi2test'

maxlag = 2

def grangers_causality_matrix(data, laenge, test = 'ssr_chi2test', verbose=False):
    
    dataset = pd.DataFrame(np.zeros((len(laenge), len(laenge))), columns=laenge, index=laenge)
    for c in dataset.columns:
        for r in dataset.index:
            data = pd.concat([df[c],df[r]], axis = 1)
            test_result = sm.tsa.stattools.grangercausalitytests(data,2, addconst=True, verbose=True)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value

    dataset.columns = [var for var in ticker_new]

    dataset.index = [var for var in ticker_new]
    
    dataset.insert(loc=0, column= 'Causality', value= ticker_new)
    
    return dataset

dataset_2 = pd.DataFrame(grangers_causality_matrix(dataset, laenge = dataset.columns))
dataset_2.to_excel(writer,'Granger Causality', index=False)
writer.save()

# Upload Values to SQLite Database
db_file = 'AQM_Group_F_Crypto.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS Granger_Causality             (Statistics, BTC Real,LTC Real,ETH Real, BCH Real, XRP Real, EUR Real, CHF Real, GBP Real, JPY Real,CNY Real, CAD Real, AUD Real, MXN Real, CHFEURR Real, GBPEURR Real, JPYEURR Real, CNYEURR Real, CADEURR Real, AUDEURR Real, MXNEURR Real)')

dataset_2 = dataset_2.values.tolist()

c.executemany("INSERT OR REPLACE INTO Granger_Causality                 (Statistics, BTC ,LTC ,ETH , BCH , XRP, EUR, CHF, GBP, JPY, CNY, CAD, AUD, MXN, CHFEURR, GBPEURR, JPYEURR, CNYEURR, CADEURR, AUDEURR, MXNEURR) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", dataset_2)

conn.commit()
c.close()
conn.close()

print('\nThe results have been saved to an Excel File and SQLite Database.')

