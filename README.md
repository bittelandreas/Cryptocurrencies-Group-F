Group F: Cryptocurrencies

# General obligatory tasks for each module thesis
1. Prepare a report structure in line with the specification below:
    a. Short literature review, use Zotero for methodology
    b. Review of methodology
    c. Description of data, sources, etc.
    d. Description of the database setup
    e. Model description
    f. Descriptive statistics
    g. Main Model Results
    h. Conclusions
2. Review the methodology and a few research papers.
3. Download data using your Eikon access in Excel or Bloomberg access in Excel. Do not download data manually. Document your work.
4. As an additional check, use Python-based program for Refinitiv API or Bloomberg API to re-download the data and verify whether the resulting data is identical. Document your work.
5. Verify and clean the data. Look for missing or wrong data, anomalies, etc.
6. Run initial analysis in Python. You can complement it using Pivots.
7. Create a SQL database and upload clean data to the database.
8. Run the analysis in Jupyter Notebook, including plotting and normality tests.
9. Export the results to the SQL database that you created for this project.
10. Summarize and visualize the results in clear and concise manner.
11. Save charts and use Python to export tables results to excel/word. Integrate them into the project.
12. Create a repository on GitHub or Google Collaboratory. Upload a small piece of your notebook and a small dataset, make sure that the notebook commands are executed in the cloud. Provide the link and a snapshot.
(13. Shiny. Run the program from the class on your device. Try to visualize your own data, or alternatively, data from the class. Provide a snapshot from the functional output website.) - gelöscht

# Required Project Deliverables to be uploaded on Moodle:
1. Written report in word or pdf including
2. Code in Jupyter Notebook
3. Database (if smaller than 10 MB, otherwise a few snapshots of key tables)
4. All the files need to be zipped

# Data:
2. For cryptocurrencies, using Bitcoin, Ethereum and others


# What drives asset prices? Causality and cointegration
• Download daily data for asset prices for at least 20 assets
• Test for stationarity in level and first difference, up to 2 lags
• For each asset (as dependent variable), create a loop and test the relationship with all other assets, in pairs:
    o Run the cointegration test for up to 2 lags table
    o Run the granger causality test for and verify which asset is leading
    o For each regression and test save the results into a table

Output example: cointegration test results for German bonds vs. US Treasuries, German bonds vs. French bonds, French bonds and US Treasuries, etc.
