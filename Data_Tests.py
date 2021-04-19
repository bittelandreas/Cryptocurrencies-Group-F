#!/usr/bin/env python
# coding: utf-8

# # What Are the Commonly Used Statistical Tests in Data Science

# Business analytics and data science is a convergence of many fields of expertise. Professionals form multiple domains and educational backgrounds are joining the analytics industry in the pursuit of becoming data scientists.
# 
# Two kinds of data scientist I met in my career. One who provides attention to the details of the algorithms and models. They always try to understand the mathematics and statistics behind the scene. Want to take full control over solution and the theory behind it. The other kind are more interested in the end result without looking at the theoretical details. They are fascinated by the implementation of new and advanced models. Inclined towards solving the problem in hand rather than the theory behind the solution.
# 
# Believers of both of these approaches have their own logic to support their stand. I respect their choices.
# 
# In this post, I shall share some statistical tests that are commonly used in data science. It will be good to know some of these irrespective of the approach you believe in.
# 
# In statistics, there are two ways of drawing an inference from any exploration. Estimation of parameters is one of the ways. Here unknown values of population parameters are computed through various methods. The other way is testing of hypothesis. It helps us to test the parameter values that are guessed by some prior knowledge.
# 
# I shall list out some statistical test procedures which you will frequently encounter in data science.

# ### As a data scientist, do I really need to know hypothesis testing?

# In most decision-making procedures in data science, we are knowing or unknowingly using hypothesis testing. Here are some evidences in support of my statement.
# 
# Being data scientist, the kind of data analysis we do can be segregated into four broad areas —
# 
# 1. Exploratory Data Analysis (EDA)
# 
# 2. Regression and Classification
# 
# 3. Forecasting
# 
# 4. Data Grouping
# 
# Each of these areas include some amount of statistical testing.

# # Exploratory Data Analysis (EDA)

# It is an unavoidable part of data science in which every data scientist spends a significant amount of time. It establishes the foundation for creating machine learning and statistical models. Some common tasks that involve statistical testing in EDA are —
# 
# 1. Test for normality
# 
# 2. Test for Outliers
# 
# 3. Test for correlation
# 
# 4. Test of homogeneity
# 
# 5. Test for equality of distribution
# 
# Each of these tasks involves testing of hypothesis at some point.

# ## Test for normality

# Normality is everywhere in Statistics. Most theories we use in statistics are based on normality assumption. Normality means the data should follow a particular kind of probability distribution, which is the normal distribution. It has a particular shape and represented by a particular function.
# 
# In Analysis of Variance(ANOVA), we assume normality of the data. While doing regression we expect the residual to follow normal distribution.
# 
# To check normality of data we can use Shapiro–Wilk Test. The null hypothesis for this test is — the distribution of the data sample is normal.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import numpy as np
from scipy import stats


df = stats.norm.rvs(loc = 2.5, scale = 2, size = 100)
shapiro_test = stats.shapiro(df)

print(shapiro_test)


# In[ ]:





# ## Test for Outliers

# When I start any new data science use case, where I have to fit some model, one of the routine tasks I do is detection of outliers in the response variable. Outliers affect the regression models greatly. A careful elimination or substitution strategy is required for the outliers.
# 
# An outlier can be global outlier if its value significantly deviate from rest of the data. It is called contextual outlier if it deviates only from the data point originated from a particular context. Also, a set of data point can be collectively outlier when they deviate considerably from the rest.
# 
# The Tietjen-Moore test is useful for determining multiple outliers in a data set. The null hypothesis for this test is — there are no outliers in the data.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import scikit_posthocs

x = np.array([-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06, 0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01])
scikit_posthocs.outliers_tietjen(df, 2) #ursprünglich war in der Klammer (x,2)


# In[ ]:





# In[ ]:





# ## Test for correlation

# In data science, we deal with a number of independent variables that explain the behavior of the dependent variable. Significant correlation between the independent variables may affect the estimated coefficient of the variables. It makes the standard error of the regression coefficients unreliable. Which hurts the interpretability of the regression.
# 
# When we calculate the correlation between two variables, we should check the significance of the correlation. It can be checked by t-test. The null hypothesis of this test assumes that the correlation among the variables is not significant.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy.stats import pearsonr

data1 = stats.norm.rvs(loc=3, scale=1.5, size=20)
data2 = stats.norm.rvs(loc=-5, scale=0.5, size=20)
stat, p = pearsonr(data1, data2)
print(stat, p)


# In[ ]:





# ## Test of homogeneity

# It would be convenient to explain the test of homogeneity if I use an example. Suppose you we want to check if the viewing preference of Netflix subscribers are same for males and females. You can use Chi-square test for homogeneity for the same. You have to check whether the frequency distribution of the males and females are significantly different from each other.
# 
# The null hypotheses for the test is the two data sets are homogeneous.

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

# Sometimes in data analysis we require checking if the data follows a particular distribution. Even we may want to check if two samples follow the same distribution. In such cases we use Kolmogorov-Smirnov (KS) test. We often use KS test to check for goodness of fit of a regression model.
# 
# This test compares the empirical cumulative distribution functions (ECDF) with the theoretical distribution function. The null hypothesis for this test assumes that the given data follows the specified distribution.

# In[120]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy import stats

x = np.linspace(-25, 17, 6)
stats.kstest(df, 'norm') #ursprünglich war in der Klammer (x,'norm')


# In[ ]:





# # Regression and Classification

# Most of the modeling we do in data science fall under either regression or classification. Whenever we predict some value or some class, we take help of these two methods.
# 
# Both regression and classification involves statistical tests at different stages of decision making. Also, the data need to satisfy some prerequisite conditions to be eligible for these tasks. Some tests are required to be performed to check these conditions.
# 
# Some common statistical tests associated with regression and classification are —
# 
# 1. Test for heteroscedasticity
# 
# 2. Test or multicollinearity
# 
# 3. Test of the significance of regression coefficients
# 
# 4. ANOVA for regression or classification model

# ## Test for heteroscedasticity

# Heteroscedasticity is a quite heavy term. It simply means unequal variance. Let me explain it with an example. Suppose you are collecting income data from different cities. You will see that the variation of income differs significantly over cities.
# 
# If the data is heteroscedastic, it affects the estimation of the regression coefficients largely. It makes the regression coefficients less precise. The estimates will be far from actual values.
# 
# To test heteroscedasticity in the data White’s Test can be used. White’s test considers the null hypothesis — the variance is constant over the data.

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

# Data science problems often include multiple explanatory variables. Some time these variables become correlated due to their origin and nature. Also, sometimes we create more than one variable from the same underlying fact. In these cases the variables become highly correlated. It is called multicollinearity.
# 
# Presence of multicollinearity increases standard error of the coefficients of the regression or classification model. It makes some important variables insignificant in the model.
# 
# Farrar–Glauber Test can be used to check the presence of multicollinearity in the data.

# In[11]:


# no code


# In[ ]:





# ## Test of the significance of regression coefficients

# In classification or regression models we require identifying the important variables which have strong influence on the target variable. The models perform some tests and provide us with the extent of significance of the variables.
# 
# t-test is used in models to check the significance of the variables. The null hypothesis of the test is- the coefficients are zero. You need to check p-values of the tests to understand the significance of the coefficients.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy import stats

rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
stats.ttest_1samp(rvs1, 7)


# In[ ]:





# ## ANOVA for regression or classification model

# While developing regression or classification model, we perform Analysis of Variance (ANOVA). It checks the validity of regression coefficients. ANOVA compares the variation due to model with the variation due to error. If the variation due to model is significantly different from variation due to error, the effect of the variable is significant.
# 
# F-test is used to take the decision. The null hypothesis in this test is — the regression coefficient is equal to zero.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import scipy.stats as stats

data1 = stats.norm.rvs(loc = 3, scale = 1.5, size = 20)
data2 = stats.norm.rvs(loc = -5, scale = 0.5, size = 20)
stats.f_oneway(data1,data2)


# In[ ]:





# # Forecasting

# In data science we deal with two kinds of data- cross-section and time series. The profiles of a set of customers on an e-commerce website are a cross-section data. But, the daily sales of an item in the e-commerce website for a year will be time series data.
# 
# We often use forecasting models on time series data to estimate the future sales or profits. But, before forecasting, we go through some diagnostic checking of the data to understand the data pattern and its fitness for forecasting.
# 
# As a data scientist I frequently use these tests on time series data:
# 
# 1. Test for trend
# 
# 2. Test for stationarity
# 
# 3. Test for autocorrelation
# 
# 4. Test for causality
# 
# 5. Test for temporal relationship

# ## Test for trend

# Data generated over time from business often shows an upward or downward trend. Be it sales or profit or any other performance metrics that depicts business performance, we always prefer to estimate the future movements.
# 
# To forecast the such movements, you need to estimate or eliminate the trend component. To understand if the trend is significant, you can use some statistical test.
# 
# Mann-Kendall Test can be used to test the existence of trend. The null hypothesis assumes that there is no significant trend.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

pip install pymannkendall

import numpy as np
import pymannkendall as mk

data = np.random.rand(250,1)
test_result = mk.original_test(data)
print(test_result)


# In[ ]:





# ## Test for stationarity

# Non-stationarity is an inherent characteristic of most time series data. We always need to test for stationarity before any time series modeling. If the data is non-stationary it may produce unreliable and spurious results after modeling. It will lead to a poor understanding of the data.
# 
# Augmented Dickey-Fuller (ADF) can be used to check for non-stationarity. The null hypothesis for ADF is the series is non-stationary. At 5% level of significance, if the p-value is less than 0.05, we reject the null hypothesis.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from statsmodels.tsa.stattools import adfuller

X = [15, 20, 21, 20, 21, 30, 33, 45, 56]
result = adfuller(X)
print(result)


# In[ ]:





# ## Test for autocorrelation

# For time series data, the causal relationship between past and present values is a common phenomenon. For financial time series often we see that current price is influenced by the prices of the last few days. This feature of time series data is measured by autocorrelation.
# 
# To know whether the autocorrelation is strong enough, you can test for it. Durbin-Watson test reveals the extent of it. The null hypothesis for this test assumes that there is no autocorrelation between the values.

# In[13]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from statsmodels.stats.stattools import durbin_watson

X = [15, 20, 21, 20, 21, 30, 33, 45, 56]
result = durbin_watson(X)
print(result)


# In[ ]:





# ## Test for causality

# Two time series variable can share causal relationship. If you are familiar with financial derivatives, a financial instrument defined on underlying stocks, you would know that spot and future values have causal relationships. They influence each other according to the situation.
# 
# The causality between two variables can be tested by Granger Causality test. This test uses a regression setup. The current value of one variable regresses on lagged values of the other variable along with lagged values of itself. The null hypothesis of no causality is determined by F-test.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

data = sm.datasets.macrodata.load_pandas()
data = data.data[[“realgdp”, “realcons”]].pct_change().dropna()
gc_res = grangercausalitytests(data, 4)


# In[ ]:





# ## Test for temporal relationship

# Two time series sometimes moves together over time. In the financial time series you will often observe that spot and future price of derivatives move together.
# 
# This co-movements can be checked through a characteristic called cointegration. This cointegration can be tested by Johansen’s test. The null hypothesis of this test assumes no cointegartion between the variables.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from statsmodels.tsa.vector_ar.vecm import coint_johansen

data = sm.datasets.macrodata.load_pandas()
data = data.data[[“realgdp”, “realcons”]].pct_change().dropna()
#x = getx() # dataframe of n series for cointegration analysis
jres = coint_johansen(data, det_order=0, k_ar_diff=1

print(jres.max_eig_stat)
print(jres.max_eig_stat_crit_vals)


# In[ ]:





# # Data Grouping

# Many times in real-life scenario we try to find similarity among the data points. The intention becomes grouping them together in some buckets and study them closely to understand how different buckets behave.
# 
# The same is applicable for variables as well. We identify some latent variable those are formed by the combination of a number of observable variables.
# 
# A retail store might be interested to form segments among its customers like — cost-conscious, brand-conscious, bulk-purchaser, etc. It requires grouping of the customers based on their characteristics like — transactions, demographics, psychographics, etc.
# 
# In this area we often encounter the following tests:
# 
# 1. Test of sphericity
# 
# 2. Test for sampling adequacy
# 
# 3. Test for clustering tendency

# ## Test of sphericity

# If the number of variables in the data is very high, the regression models in this situation tend to perform badly. Besides, identifying important variables becomes challenging. In this scenario, we try to reduce the number of variables.
# 
# Principal Component Analysis (PCA) is one method of reducing the number of variables and identifying major factors. These factors will help you built a regression model with reduced dimension. Also, help to identify key features of any object or incident of interest.
# 
# Now, variables can form factors only when they share some amount of correlation. It is tested by Bartlet’s test. The null hypothesis of this test is — variables are uncorrelated.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from scipy.stats import bartlett

a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]

stat, p = bartlett(a, b, c)
print(p, stat)


# In[ ]:





# ## Test for sampling adequacy

# The PCA method will produce a reliable result when the sample size is large enough. This is called sampling adequacy. It is to be checked for each variable.
# 
# Kaiser-Meyer-Olkin (KMO) test is used to check sampling adequacy for the overall data set. The statistic measures the proportion of variance among variables that could be common variance.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo


a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]

df = pd.DataFrame({‘x’:a,’y’:b,’z’:c})
kmo_all,kmo_model=calculate_kmo(df)

print(kmo_all,kmo_model)


# ## Test for clustering tendency

# To group the data in different buckets, we use clustering techniques. But before going for clustering you need to check if there is clustering tendency in the data. If the data has uniform distribution then it not suitable for clustering.
# 
# Hopkins test can check for spatial randomness of variables. Null hypothesis in this test is — the data is generated from non-random, uniform distribution.

# In[ ]:


# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

from sklearn import datasets
from pyclustertend import hopkins
from sklearn.preprocessing import scale

X = scale(datasets.load_iris().data)
hopkins(X,150)


# In[ ]:




