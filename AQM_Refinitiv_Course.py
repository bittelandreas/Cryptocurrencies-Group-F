# Get assets under management for each investor in Apple stock.
# Advanced Quantitative Methods
# Demo Code for finding "Assets under Management"
# of Investors of AAPL -- this is a real-world case based on our work


# Step 0 : Fire up Refinitiv Workspace

# Step 1a : Get code for stock AAPL.O

# Step 1b : Use Data-Browser to find series code
# (Show data filter and browser in the tap)

'TR.OwnTurnover'

# Step one Import the libraries
# this is the Eikon library, almost all APIs have this type of class that allows one to Specify how to interact with the API
import eikon as ek # pip install eikon
import pandas as pd
import numpy as np
# Handles data
import datetime


# Here you can put your key; I read from a local file to avoid sharing password
# creds = 'xudsan;ljdnsadlf'kdamnskadms'
creds = pd.read_csv('~/spark_pw.csv')
# Tell EIkon what the key is
ek.set_app_key(creds.iloc[2,1])


# Define a wrapper function (not strictly necessary)
def holding_data(instrument='AAPL.O', fields=[], start_year=2019, end_year=2020):
    fields = ["TR.InvestorFullName",
              "TR.InvParentType",
              "TR.InvestorRegion",
              "TR.FundInvestorType",
              "TR.InvestorType",
              "TR.SharesHeld.Date",
              "TR.SharesHeld",
              "TR.SharesHeldValue",
              "TR.SharesHeldValChg",
              # Perhaps also of use to Tamasz
              'TR.InvestorTotalAssets',
              'TR.TotalEquityAssets',
              'TR.InvInvmtOrientation',
              'TR.InvInvestmentStyleCode',
              'TR.OwnTurnover']
    dat, err = ek.get_data('AAPL.O',
                           fields = fields,
                           parameters={'SDate': '{}-01-01'.format(2019),
                                       'EDate': '{}-12-31'.format(2020),
                                       'Frq':'Q'} # Y # D #M
                           )

    dat['Date'] = pd.to_datetime(dat['Date'], format='%Y-%m-%d').dt.date # convert format to %Y-%m-%d
    dat = dat.sort_index() # sort Date in ascending order
    # data base containts weak data points: drop na's
    dat = dat.dropna() # I'm ambivalent about this...

    return(dat)


# Assigns the return of the function to a variable
dat = holding_data()


print(dat.head(10))
# Writes the data frame to the CSV


# Composition of the portfolio for that period
dat.to_csv('investor.csv')

