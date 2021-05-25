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

def OLS(y,x):
        
    #Beta estimates [(X'X)^-1]X'Y
    x_xt= np.dot(x.T,x)         # multiply vector X with transposed vector X'
    x_x_inv=np.linalg.inv(x_xt)      # invert the matrix

    x_y= np.dot(x.T,y)                 # multiply transposed vector X' with Y
    est_betas= np.dot(x_x_inv,x_y)  # multiply (X'X)^-1 with X'Y
    est_Y=      np.dot(est_betas,x.T)  # multiply est_betas with X'
        
    nb_obs=x.shape[0]  
    rank=x.shape[1] #equals p = number of regressors 
    deg_freedom_residual=nb_obs-rank
    
    #Y estimate
    est_y = np.dot(x,est_betas)
    est_resid = y-est_y
    
    est_resid_2 = np.dot(est_resid.T,est_resid)
    
    est_sigma_2= est_resid_2 / nb_obs
    
    est_var_covar_beta = np.dot(x_x_inv,est_sigma_2)  #Variance-Covariance Matrix of Betas
    
    est_var_beta = np.diag(est_var_covar_beta)
    est_sd_beta=np.sqrt(est_var_beta)    # standard errors of beta
    t_stats= est_betas/est_sd_beta       # t-stats
    pvals = stats.t.sf(np.abs(t_stats), nb_obs-1)*2
  
    SST = np.var(y)*nb_obs
    SSR = est_resid_2
    R_2 = 1-(SSR/SST)
    
    llf = (-0.5*nb_obs * np.log(2 * np.pi*est_sigma_2)) -(0.5*nb_obs*(1/np.log(est_sigma_2))*est_resid_2)
    
    OLS_results = {
        'Y': y,
        'est_Y': est_Y,
        'X': x,
        'est_betas': est_betas,
        'est_resid': est_resid,
        'nb_obs': nb_obs,
        'deg_freedom_residual': deg_freedom_residual,
        'est_sigma_2': est_sigma_2,
        'est_var_covar_beta': est_var_covar_beta,
        't_stats': t_stats,
        'pvals': pvals,
        'llf': llf,
        'nb_parms': rank,
        'R_2': R_2
    }
                                                          
    return OLS_results