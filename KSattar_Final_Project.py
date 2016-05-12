# LendingClub Loan Performance
# EECS E6895, Advanced Big Data Analytics
# Kaivan K. Sattar, 5/10/2016

import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy, time, warnings, os, requests, zipfile, io, pickle
import matplotlib.pyplot as plt

from sklearn import tree, linear_model, neighbors
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from fredapi import Fred

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions

def download_GDELT(startt, stopp, dates, direc):    
    for d in dates:
        try:
            r = requests.get('http://data.gdeltproject.org/gkg/' + d + '.gkg.csv.zip')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(direc)
            
        except:
            continue


def process_GDELT(sc, direc, date, states):
    warnings.filterwarnings('ignore')
    os.chdir(direc)
    
    sqlContext = SQLContext(sc)
    news = sc.textFile(direc+date+'.gkg.csv')
    header = news.first()
    fields = [StructField(field_name, StringType(), True) for field_name in header.split('\t')]
    
    # only first and second fields are not strings, everything else is StringType by default
    fields[1].dataType = IntegerType()
    schema = StructType(fields)
    
    # identify header row
    newsHeader = news.filter(lambda l: 'DATE\tNUMARTS\t' in l)
    newsHeader.collect()
    newsNoHeader = news.subtract(newsHeader)
    
    # split columns at \t, set column types, convert to DF
    news_df = newsNoHeader.map(lambda k: k.split('\t')).map(lambda p: (p[0], int(p[1]), p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10])).toDF(schema)
    news_df_pd = news_df.toPandas()
    
    news_df_pd['TONE_num'] = news_df_pd.TONE.str.split(',').str[0].astype(float)
    news_df_pd['loc_1'] = news_df_pd['LOCATIONS'].str.split('#').str.get(2)
    news_df_pd['loc_2'] = news_df_pd['LOCATIONS'].str.split('#').str.get(8)
    news_df_pd['loc_3'] = news_df_pd['LOCATIONS'].str.split('#').str.get(14)
    
    news_df_pd = news_df_pd[(news_df_pd['loc_1'] == 'US') | (news_df_pd['loc_2'] == 'US') | (news_df_pd['loc_3'] == 'US')]
    news_df_pd['st_1'] = news_df_pd['LOCATIONS'].str.split('#').str.get(3).str[-2:]
    news_df_pd['st_2'] = news_df_pd['LOCATIONS'].str.split('#').str.get(9).str[-2:]
    news_df_pd['st_3'] = news_df_pd['LOCATIONS'].str.split('#').str.get(15).str[-2:]
    
    tones = {}
    
    for s in states:
        temp = news_df_pd[(news_df_pd['st_1'] == s) | (news_df_pd['st_2'] == s) | (news_df_pd['st_3'] == s)]
        tones[s] = temp['TONE_num'].mean()
        
    return tones
    

def fred_data():
    fred = Fred(api_key='3f540e789def0dd6cdc33437537f1b63')
    states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
    fred_df = pd.DataFrame()
    
    for s in states:
        state_df = pd.DataFrame(pd.date_range(start='2006-12-01', end='2016-05-01', freq='M').shift(1, freq=pd.datetools.day))
        state_df.columns = ['date']
        
        for var in [s+'UR', s+'STHPI', 'MEHOINUS'+s+'A672N']:
            var_df = pd.DataFrame(fred.get_series(var)).reset_index()
            var_df['state'] = s
            
            if var == 'MEHOINUS'+s+'A672N':
                var_df['inc_D1_L2'] = var_df[0].shift(2) - var_df[0].shift(3)
                var_df['inc_D3_L2'] = var_df[0].shift(2) - var_df[0].shift(5)
                var_df['inc_L2'] = var_df[0].shift(2)
                var_df.columns = ['date', 'median_income', 'state', 'inc_D1_L2', 'inc_D3_L2', 'inc_L2']
                
            elif var == s+'STHPI':
                var_df['HPI_D1_L2'] = var_df[0].shift(2) - var_df[0].shift(3)
                var_df['HPI_D3_L2'] = var_df[0].shift(2) - var_df[0].shift(5)
                var_df['HPI_L2'] = var_df[0].shift(2)
                var_df.columns = ['date', 'HPI', 'state', 'HPI_D1_L2', 'HPI_D3_L2', 'HPI_L2']
                
            elif var == s+'UR':
                var_df['unemp_D1_L2'] = var_df[0].shift(2) - var_df[0].shift(3)
                var_df['unemp_D3_L2'] = var_df[0].shift(2) - var_df[0].shift(5)
                var_df['unemp_L2'] = var_df[0].shift(2)
                var_df.columns = ['date', 'unemp', 'state', 'unemp_D1_L2', 'unemp_D3_L2', 'unemp_L2']
                
            state_df = pd.merge(state_df, var_df, how='outer', on='date')
        
        state_df = state_df[['date','state_x','unemp','unemp_D1_L2','unemp_D3_L2','unemp_L2','HPI','HPI_D1_L2','HPI_D3_L2','HPI_L2','median_income','inc_D1_L2','inc_D3_L2','inc_L2']]
        state_df.columns = ['date','state','unemp','unemp_D1_L2','unemp_D3_L2','unemp_L2','HPI','HPI_D1_L2','HPI_D3_L2','HPI_L2','median_income','inc_D1_L2','inc_D3_L2','inc_L2']
        state_df[['date','state','HPI','HPI_D1_L2','HPI_D3_L2','HPI_L2','median_income','inc_D1_L2','inc_D3_L2','inc_L2']] = state_df[['date','state','HPI','HPI_D1_L2','HPI_D3_L2','HPI_L2','median_income','inc_D1_L2','inc_D3_L2','inc_L2']].ffill()
        
        fred_df = fred_df.append(state_df)
    
    fred_df = fred_df[(fred_df['date'].dt.day == 1) & (fred_df['date'].dt.year >= 2007)].sort_values(['state','date'])
    
    return fred_df

# if this doesn't work, please download data from www.lendingclub.com/info/download-data.action
# loan data, not declined loan data, from 2007-2015
def download_LC(direc):
    for file in ['a','b','c','d']:
        try:
            r = requests.get('https://resources.lendingclub.com/LoanStats3' + file + '.csv.zip')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(direc)
        
        except:
            continue


# Data cleaning and variable creation
def LC_data(direc, startt, stopp, tone_of_tones, states):
    df = pd.read_csv(direc+'LoanStats3a.csv', header=1, low_memory=False)
    
    for cohort in ['b','c','d']:
        df = df.append(pd.read_csv(direc+'LoanStats3'+cohort+'.csv', header=1, low_memory=False))

    # For consistency, only consider 'finished' loans. Loans that are still in
    # progress (Current/Delinquent etc.) are not directly comparable to loans
    # that have either fully paid off or charged off
    
    df = df[(df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off')]
    df['paid'] = (df['loan_status'] == 'Fully Paid').astype(int)
    df['len_title'] = df.title.str.len()
    
    df['revol_util'] = df.revol_util.str.split('%').str.get(0).astype(float)
    df['int_rate'] = df.int_rate.str.split('%').str.get(0).astype(float)
    #df['emp_length'] = pd.to_numeric(pd.core.strings.str_strip(df['emp_length'].str.replace(r'[$,<+n/years]', '')), errors='coerce')
    
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'])
    df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'])
    df['len_cr_history'] = (df.issue_d - df.earliest_cr_line) / np.timedelta64(1,'D')
    
    df['lfrac_inc'] = df.loan_amnt / df.annual_inc
    df['ifrac_inc'] = df.installment / df.annual_inc
    df['lfrac_inc'][df['lfrac_inc'] == np.inf] = np.nan
    df['ifrac_inc'][df['ifrac_inc'] == np.inf] = np.nan
    
    # construct monthly average of TONE by state
    # {datestr : {state : TONE}}
    dates = pd.DataFrame(pd.date_range(start=startt, end=stopp, freq=pd.tseries.offsets.DateOffset(months=1)))
    dates.columns = ['date']
    dates['date'] = dates['date'].dt.date.astype(str).str[:-3]
    dates = list(dates.values.flatten())
    
    tone_df = pd.DataFrame(np.nan, index=[0], columns=['state','TONE','date'])
    
    if len(tone_of_tones) > 0:
        for k1,v1 in tone_of_tones.items():
            strr = k1[:4] + '-' + k1[4:6] + '-' + k1[6:8]
            temp = pd.DataFrame.from_dict(v1, orient='index').reset_index()
            temp['ym'] = strr
            temp.columns = ['state','TONE','date']
            tone_df = tone_df.append(temp)
        
        tone_df = tone_df[tone_df['date'].isnull() == False].sort_values('date')
        tone_df['ym'] = tone_df['date'].str.split('-').str.get(0) + '-' + tone_df['date'].str.split('-').str.get(1)
        tone_df_2 = tone_df.groupby(['ym','state'])['TONE'].mean().reset_index()
        df['ym'] = (df['issue_d']).dt.date.astype(str).str[:7]
        df['state'] = df['addr_state']
        df = pd.merge(df, tone_df_2, how='outer', on=['ym','state'])
    
    return df


# Run a simple logit model - just quickly testing ideas
# Significant variables: term, len_title, lfrac_inc, dti, inq_last_6mths,
# grade, sub_grade, revol_bal, revol_util, total_acc, open_acc, len_cr_history

# Essentially, I used this to build a heuristic feature selection

def run_logit(spec, df):
    #df[df == np.inf] = np.nan
    df[df.isnull() == True] = np.nan
    Y,X = patsy.dmatrices(spec, df, return_type='dataframe')
    print(sm.Logit(Y,X).fit().summary())

# This helper function fits a model, calculates the error, and records runtime

def update_dict(model, name, errors, train_X, train_Y, test_X, test_Y, sim):
    start = time.clock()
    yhat = model.fit(train_X, train_Y).predict(test_X)
    score = roc_auc_score(test_Y, yhat)
    end = time.clock()
    runtime = end-start

    if name in errors:
        errors.update({name : (round(errors[name][0] + (score/sim), 5), round(errors[name][1] + runtime, 5))})
    else:
        errors.update({name : (round(score/sim, 5), runtime)})

    return errors

# 1 = pick a classification
# 2 = divide into training/test
# 3 = simulate multiple times and see what wins

def model_selection(sim, spec, df, search):
    if search == 'breadth':
        classif = [[linear_model.LogisticRegression(),'Logit'],[LinearSVC(),'LinearSVC'],[neighbors.KNeighborsClassifier(1),'KNN1'],[neighbors.KNeighborsClassifier(5),'KNN5'],[neighbors.KNeighborsClassifier(10),'KNN10'],[neighbors.KNeighborsClassifier(20),'KNN20'],[neighbors.KNeighborsClassifier(50),'KNN50'],[RandomForestClassifier(n_estimators=10, max_depth=1),'RF10_1'],[RandomForestClassifier(n_estimators=10, max_depth=2),'RF10_2'],[RandomForestClassifier(n_estimators=30, max_depth=1),'RF30_1'],[RandomForestClassifier(n_estimators=30, max_depth=2),'RF30_2'],[RandomForestClassifier(n_estimators=100, max_depth=1),'RF100_1'],[RandomForestClassifier(n_estimators=100, max_depth=2),'RF100_2'],[RandomForestClassifier(n_estimators=150, max_depth=1),'RF150_1'],[RandomForestClassifier(n_estimators=150, max_depth=2),'RF150_2'],[tree.DecisionTreeClassifier(min_samples_split=5),'DT5'],[tree.DecisionTreeClassifier(min_samples_split=10),'DT10'],[tree.DecisionTreeClassifier(min_samples_split=15),'DT15'],[tree.DecisionTreeClassifier(min_samples_split=20),'DT20'],[tree.DecisionTreeClassifier(min_samples_split=30),'DT30'],[tree.DecisionTreeClassifier(min_samples_split=50),'DT50']]
    
    elif search == 'depth':
        classif = [[tree.DecisionTreeClassifier(min_samples_split=50, max_depth=10),'DT50_10'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=20),'DT50_20'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=30),'DT50_30'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=40),'DT50_40'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=50),'DT50_50'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=60),'DT50_60'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=70),'DT50_70'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=80),'DT50_80'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=90),'DT50_90'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=100),'DT50_100'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=110),'DT50_110'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=120),'DT50_120'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=130),'DT50_130'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=140),'DT50_140'],
                [tree.DecisionTreeClassifier(min_samples_split=50, max_depth=150),'DT50_150']]
    
    errors = {}

    Y,X = patsy.dmatrices(spec, df)
    X_df = pd.DataFrame(X, columns=X.design_info.column_names)
    Y_num = np.ravel(pd.DataFrame(Y, columns=Y.design_info.column_names))

    for k in range(sim):
        # Randomly select 90% of sample as training data
        rows = np.random.choice(X_df.index.values, int(len(X_df)*0.9), replace=False)
        train_X = X_df.ix[rows]
        test_X = X_df.drop(rows)
        train_Y = Y_num[rows]
        test_Y = np.delete(Y_num, rows)

        # Fit and calculate error of all candidate models
        for j in range(len(classif)):
            errors = update_dict(classif[j][0], classif[j][1], errors, train_X, train_Y, test_X, test_Y, sim)
            print('Simulation = ', k+1, ', model = ', classif[j][1])

    winner = max(errors, key=errors.get)
    print('Winner = ', winner)
    return errors, winner

def main():
    pd.set_option('display.float_format', lambda x:'%f'%x)
    warnings.filterwarnings('ignore')
    
    ### SET PARAMETERS ###
    direc = '/users/kaivansattar/desktop/misc/'
    download_GDELT = False
    process_GDELT = False
    download_LC = True
    process_LC = False
    startt = '2013-04-01'
    stopp = '2016-05-08'
    
    states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
    
    # create list of dates to pull GDELT data for
    dates = pd.DataFrame(pd.date_range(start=startt, end=stopp, freq='D'))
    dates.columns = ['date']
    dates['date'] = dates['date'].dt.date.astype(str).str.replace('-','')
    dates = list(dates.values.flatten())
    
    # download daily GDELT data (100 gb for 2013-04-01 to 2016-05-08)
    if download_GDELT == True:
        download_GDELT(startt, stopp, dates, direc+'GDELT_data/')
    
    # process this data using Spark
    tone_of_tones = {}
    
    if process_GDELT == True:
        for d in dates:
            tone_of_tones[d] = process_GDELT(sc, direc+'GDELT_data/', d, states)
        
        pickle.dump(tone_of_tones, open(direc + 'GDELT_data/tone_of_tones.p', 'wb'))
    
    else:
        tone_of_tones = pickle.load(open(direc + 'GDELT_data/tone_of_tones.p', 'rb'))
    
    # pull economic indicators from St. Louis FRED
    fred_df = fred_data()
    
    # download LendingClub data (~700 mb for 2007-2015 data)
    # if this doesn't work, please download data from www.lendingclub.com/info/download-data.action
    # loan data, not declined loan data, from 2007-2015
    if download_LC == True:
        download_LC(direc)
    
    # process LendingClub data
    if process_LC == True:
        LC_df = LC_data(direc, startt, stopp, tone_of_tones, states)
        pickle.dump(LC_df, open(direc + 'LC_df.p', 'wb'))
    
    else:
        LC_df = pickle.load(open(direc + 'LC_df.p', 'rb'))
    
    # merge FRED and LendingClub data
    df = pd.merge(LC_df, fred_df, how='inner', left_on=['issue_d','addr_state'], right_on=['date','state'])
    df['ratio'] = (df['annual_inc'] / df['inc_L2']).astype(float)
    
    if len(tone_of_tones) > 0:
        spec = 'paid~term+len_title+loan_amnt+lfrac_inc+ifrac_inc+dti+len_cr_history+inq_last_6mths+revol_bal+revol_util+total_acc+open_acc+sub_grade+unemp_D1_L2+unemp_L2+HPI_D1_L2+HPI_L2+inc_L2+ratio+TONE'
    else:
        spec = 'paid~term+len_title+loan_amnt+lfrac_inc+ifrac_inc+dti+len_cr_history+inq_last_6mths+revol_bal+revol_util+total_acc+open_acc+sub_grade+unemp_D1_L2+unemp_L2+HPI_D1_L2+HPI_L2+inc_L2+ratio'
    
    run_logit(spec, df)

    # 1st argument = number of simulations to run
    # 4th argument = breadth-first or depth-first model search
    # Output = (average AUC, runtime in seconds)
    errors, winner = model_selection(1, spec, df, 'breadth')
    errors, winner = model_selection(1, spec, df, 'depth')
    
    return errors, winner, df


if __name__ == "__main__": errors, winner, df = main()
