# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:14:15 2024

@author: Ion Adina Alexandra
"""


import xlwings as xw

import pandas as pd

import numpy as np


import statsmodels

from statsmodels.tsa.arima.model import ARIMA

import warnings

import matplotlib.pyplot as plt

import scipy.stats.mstats as mstats

import yfinance

import pickle

#import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
wb = xw.Book("price_dataNou.xlsx")
sheet = wb.sheets("Results")
#%%
"""
First extract the data. 

1. Transform index with dates. 
2. Remove 30th February row
3. Transform to dates the index datetimes.
4. Remove the duplicate value on the position 847, or 2020-12-10. 

0. ADDITIONAL STEP: Add also FTSE main index for potential regression. 
"""
FTSE_main = yfinance.download('^FTSE',start = '2020-09-16',end = '2024-04-23')
FTSE_data = pd.read_excel(r"C:\Users\Pupaza--DAVID\OneDrive\Desktop\LSEG\price_data.xlsx",sheet_name = "Sheet 1",usecols = 'A:G',
                          skiprows = 11)
FTSE_data = FTSE_data[::-1]
FTSE_data.index = FTSE_data.iloc[:,0]
FTSE_data = FTSE_data.drop(columns = FTSE_data.columns[0])
FTSE_data = FTSE_data.drop(index = '30-Feb-2024')
FTSE_data.index = pd.to_datetime(FTSE_data.index,format = '%Y-%m-%d')
value_recurring = list(set(FTSE_data.index))[119]
lst = list(range(941))

lst2 = list(set.difference(set(lst),set([847])))
FTSE_data = FTSE_data.iloc[lst2,:]
FTSE_data = FTSE_data.asfreq('B')
#%%
warnings.filterwarnings("ignore")
"""
1. Interpolation.
2. Add the FTSE main index closing prices and find their returns. 
"""
FTSE_data = FTSE_data.interpolate(axis = 0)
FTSE_data['FTSE close'] = FTSE_main['Close']
FTSE_data['FTSE returns'] = FTSE_main['Close'].diff()/FTSE_main['Close']
FTSE_data['FTSE returns'].fillna(method = 'ffill',inplace = True)
#%%
"""
Trend and seasonality. 
"""
def test_trends():
    """
    Determine whether there is a trend in the returns. 
    
    I use mobile moving averages of returns (20 day moving averages)
    """
    moving_averages_5D = FTSE_data['%Chg'].iloc[1:].rolling(window = 5).mean()
    moving_averages_10D = FTSE_data['%Chg'].iloc[1:].rolling(window = 10).mean()
    rolling_avg_df = pd.concat([moving_averages_5D,moving_averages_10D],axis = 1)
    rolling_avg_df.columns = ['5D MA','10D MA']
    rolling_avg_df.iloc[10:,:].plot(title = 'MA returns',grid = True)
    return rolling_avg_df

df_moving_avg = test_trends()
#%%
"""
Trend, seasonality and error of prices. 
"""
def seasonality_trend():
    """
    result: decomposition of FTSE ESG closing prices 
    
    result2: decomposition of FTSE ESG returns 
    """
    result = seasonal_decompose(FTSE_data['Close'], model='multiplicative', 
                                extrapolate_trend='freq')
    result2 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',
                                 extrapolate_trend = 'freq')
    #result2 = seasonal_decompose(FTSE_data['Close'],model = 'multiplicative',extrapolate_trend = 'freq')
    return result,result2
price_res,return_res = seasonality_trend()

#%%
def price_trends():
    """
    I find the price trends for 20 days, 5 days and 10 days rolling windows. 
    Observation:
        The default trend in Python is 5 day trend.
    """
    result1 = seasonal_decompose(FTSE_data['Close'], model='multiplicative', 
                                extrapolate_trend='freq')
    result2 = seasonal_decompose(FTSE_data['Close'],model = 'multiplicative',
                                 extrapolate_trend = 'freq',period = 20)
    result3 = seasonal_decompose(FTSE_data['Close'],model = 'multiplicative',
                                 extrapolate_trend = 'freq',period = 5)
    result4 = seasonal_decompose(FTSE_data['Close'],model = 'multiplicative',
                                 extrapolate_trend = 'freq',period = 10)
    trend1 = result1.trend
    trend2 = result2.trend
    trend3 = result3.trend
    trend4 = result4.trend
    df_trends = pd.DataFrame([trend1,trend2,trend3,trend4]).T
    df_trends.columns = ['Default','20 Day Trend','5 Day trend','10 Day trend']
    return df_trends

df_price_trends = price_trends()

#%%
def kendall_tau():
    """
    Measures the trend existence, or the correlation between TIME and RETURN values. 
    
    The value is -0.0135 for the returns, which indicates there is no clear trend in the returns. 
    
    Therefore 
    """
    concordances = sum([sum([FTSE_data['%Chg'][i]<FTSE_data['%Chg'][j] for j in range(i,939)]) 
                        for i in range(1,937)])
    discordances = sum([sum([FTSE_data['%Chg'][i]>FTSE_data['%Chg'][j] for j in range(i,939)]) 
                        for i in range(1,937)])
    kendalls_tau = (concordances-discordances)/(concordances+discordances)
    return kendalls_tau
tau = kendall_tau()
#%%
def return_trends():
    result1 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',
                                 extrapolate_trend = 'freq',period = 20)
    result2 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',
                                 extrapolate_trend = 'freq',period = 5)
    result3 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'addittive',
                                 extrapolate_trend = 'freq',period = 10)
    trend1 = result1.trend
    trend2 = result2.trend
    trend3 = result3.trend
    df_trends = pd.DataFrame([trend1,trend2,trend3]).T
    df_trends.columns = ['20 Day Trend','5 Day trend','10 Day Trend']
    return df_trends

df_return_trends = return_trends()
#%%
def return_seasonals():
    result1 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',
                                 extrapolate_trend = 'freq',period = 20)
    result2 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',
                                 extrapolate_trend = 'freq',period = 5)
    result3 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'addittive',
                                 extrapolate_trend = 'freq',period = 10)
    seasonal1 = result1.seasonal
    seasonal2 = result2.seasonal
    seasonal3 = result3.seasonal
    df_seasonal = pd.DataFrame([seasonal1,seasonal2,seasonal3]).T
    df_seasonal.columns = ['20 Day','5 Day','10 Day']
    return df_seasonal

df_return_seasonals = return_seasonals()

#%%
result1 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 5)
result2 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 10)
result3 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 20) # monthly trend-cycle
result4 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 60) # quarterly trend-cycle
result5 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 128) # semi-annual trend-cycle
result6 = seasonal_decompose(FTSE_data['%Chg'][1:],model = 'additive',period = 252) # yearly trend-cycle. 
#%%
"""
Plot the actual returns, trend, annual seasonality and ERRORS for 1 day returns of FTSE
"""
result6.plot()
plt.suptitle('Seasonal Decomposition of Daily FTSE ESG Returns')
plt.tight_layout()
plt.show()
#%%
def test_seasonality():
    df_price_res = pd.DataFrame([price_res.trend,price_res.seasonal,price_res.resid]).T
    price_res.trend.plot(title = 'Trend')
    price_res.seasonal.plot(title = 'Seasonality')
    return df_price_res
test_seasonality()
#%%

def outliers_test():
    """
    Outliers:
        1. Use winsorizing techniques for the worst 1% results and best 1% results.
        (1% quantile, 99% quantile)  
        
        2. Find the impact of winsorization on the result of ARMA(1,1)
        
        Answer to 2: There is no impact of the winsorization. 
    """
    wins_data = mstats.winsorize(FTSE_data['%Chg'][1:],limits=[0.01,0.01])
    FTSE_data['Wins chgs'] = np.insert(wins_data,0,np.nan)
    FTSE_data[['%Chg','Wins chgs']].plot(grid = True)
    arma10_model_wins = ARIMA(FTSE_data['Wins chgs'][-270:-20],order = (1,0,0)).fit()
    arma10_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (1,0,0)).fit()
    params1 = arma10_model_wins.params
    params2 = arma10_model.params
    return params1,params2
params_outliers = outliers_test()
#%%
def plot_wins_changes():
    """
    Plot winsorized changes from 17/9/2020, because the return on 16/9/2020 is NaN
    """
    FTSE_data[['%Chg','Wins chgs']].iloc[1:,:].plot(title='FTSE returns', grid=True)
    plt.ylabel('Returns')  # Adaugă eticheta axei y
    plt.show()

plot_wins_changes()


#%%
def arima_models():
    warnings.filterwarnings("ignore")
    """
    The ARIMA models for the last 270 days less the last 20 days, used for forecasting and fitting. 
    """
    arma10_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (1,0,0)).fit()
    arma20_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (2,0,0)).fit()
    arma30_model  = ARIMA(FTSE_data['%Chg'][-270:-20],order = (3,0,0)).fit()
    arma11_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (1,0,1)).fit()
    arma21_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (2,0,1)).fit()
    arima111_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (1,1,1)).fit()
    arima211_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (2,1,1)).fit()
    arima212_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (2,1,2)).fit()
    arma12_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (1,0,2)).fit()
    arma22_model = ARIMA(FTSE_data['%Chg'][-270:-20],order = (2,0,2)).fit()
    return arma10_model,arma20_model,arma30_model,arma11_model,arma21_model,arima111_model,\
            arima211_model,arima212_model,arma12_model,arma22_model

arima_mdls = arima_models()

#%%
def arima_models_alltime():
    warnings.filterwarnings("ignore")
    arma10_model = ARIMA(FTSE_data['%Chg'][:-20],order = (1,0,0)).fit()
    arma20_model = ARIMA(FTSE_data['%Chg'][:-20],order = (2,0,0)).fit()
    arma30_model  = ARIMA(FTSE_data['%Chg'][:-20],order = (3,0,0)).fit()
    arma11_model = ARIMA(FTSE_data['%Chg'][:-20],order = (1,0,1)).fit()
    arma21_model = ARIMA(FTSE_data['%Chg'][:-20],order = (2,0,1)).fit()
    arima111_model = ARIMA(FTSE_data['%Chg'][:-20],order = (1,1,1)).fit()
    arima211_model = ARIMA(FTSE_data['%Chg'][:-20],order = (2,1,1)).fit()
    arima212_model = ARIMA(FTSE_data['%Chg'][:-20],order = (2,1,2)).fit()
    arma12_model = ARIMA(FTSE_data['%Chg'][:-20],order = (1,0,2)).fit()
    arma22_model = ARIMA(FTSE_data['%Chg'][:-20],order = (2,0,2)).fit()
    return arma10_model,arma20_model,arma30_model,arma11_model,arma21_model,arima111_model,\
            arima211_model,arima212_model,arma12_model,arma22_model
            

arima_mdls_alltime = arima_models_alltime()
    
#%%
def RMSE_returns_alltime():
    """
    PE RANDAMENTE, NU PE PRETURI. 
    
    Backtest results from AR(I)MA models for the last 20 days in order to make a 
    selection of the best model. 
    
    We base the forecast on the period September 2020 and March 2024.
    """
    warnings.filterwarnings("ignore")
    forecasts_arma10 = arima_mdls_alltime[0].forecast(20)
    forecasts_arma20 = arima_mdls_alltime[1].forecast(20)
    forecasts_arma30 = arima_mdls_alltime[2].forecast(20)
    forecasts_arma11 = arima_mdls_alltime[3].forecast(20)
    forecasts_arma21 = arima_mdls_alltime[4].forecast(20)
    forecasts_arima111 = arima_mdls_alltime[5].forecast(20)
    forecasts_arima211 = arima_mdls_alltime[6].forecast(20)
    forecasts_arima212 = arima_mdls_alltime[7].forecast(20)
    forecasts_arma12 = arima_mdls_alltime[8].forecast(20)
    forecasts_arma22 = arima_mdls_alltime[9].forecast(20)
    RMSE_arma11 = np.std(np.array(forecasts_arma11)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma12 = np.std(np.array(forecasts_arma12)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma10 = np.std(np.array(forecasts_arma10)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma20 = np.std(np.array(forecasts_arma20)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma30 = np.std(np.array(forecasts_arma30) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma21 = np.std(np.array(forecasts_arma21) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arma22 = np.std(np.array(forecasts_arma22) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arima111 = np.std(np.array(forecasts_arima111)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arima211 = np.std(np.array(forecasts_arima211)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_arima212 = np.std(np.array(forecasts_arima212)-np.array(FTSE_data['%Chg'][-20:]))
    keys = ['ARMA(1,0)','ARMA(2,0)','ARMA(3,0)','ARMA(1,1)','ARMA(2,1)','ARMA(1,2)','ARMA(2,2)','ARIMA(1,1,1)',
            'ARIMA(2,1,1)','ARIMA(2,1,2)']
    values = [RMSE_arma10,RMSE_arma20,RMSE_arma30,RMSE_arma11,RMSE_arma21,RMSE_arma12,RMSE_arma22,
              RMSE_arima111,RMSE_arima211,RMSE_arima212]
    result = pd.DataFrame(dict(zip(keys,values)),index = ['RMSE'])
    result = result.T
    result['Rank'] = result['RMSE'].rank()
    return result

results_RMSE_alltime = RMSE_returns_alltime()
#%%
sheet.range("G1").value = results_RMSE_alltime
#%%

def RMSE_prices_1Y():
    """
    Forecast of prices based on 1 year training set up to 20 business days ago. 
    """
    warnings.filterwarnings("ignore")
    forecasts_arma10 = arima_mdls[0].forecast(20)
    forecasts_arma20 = arima_mdls[1].forecast(20)
    forecasts_arma30 = arima_mdls[2].forecast(20)
    forecasts_arma11 = arima_mdls[3].forecast(20)
    forecasts_arma21 = arima_mdls[4].forecast(20)
    forecasts_arma12 = arima_mdls[8].forecast(20)
    forecasts_arma22 = arima_mdls[9].forecast(20)
    forecasts_arima111 = arima_mdls[5].forecast(20)
    forecasts_arima211 = arima_mdls[6].forecast(20)
    forecasts_arima212 = arima_mdls[7].forecast(20)
    for_prices_arma10 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma10))
    for_prices_arma20 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma20))
    for_prices_arma30 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma30))
    for_prices_arma11 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma11))
    for_prices_arma21 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma21))
    for_prices_arma12 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma12))
    for_prices_arma22 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arma22))
    for_prices_arima111 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arima111))
    for_prices_arima211 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arima211))
    for_prices_arima212 = FTSE_data.iloc[-20,:]['Close']*(np.cumprod(1+forecasts_arima212))
    RMSE_prices_arma10 = np.std(np.array(for_prices_arma10)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma20 = np.std(np.array(for_prices_arma20)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma30 = np.std(np.array(for_prices_arma30)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma11 = np.std(np.array(for_prices_arma11)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma21 = np.std(np.array(for_prices_arma21)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma12 = np.std(np.array(for_prices_arma12)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arma22 = np.std(np.array(for_prices_arma22)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arima111 = np.std(np.array(for_prices_arima111)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arima211 = np.std(np.array(for_prices_arima211)-np.array(FTSE_data.iloc[-20:]['Close']))
    RMSE_prices_arima212 = np.std(np.array(for_prices_arima212)-np.array(FTSE_data.iloc[-20:]['Close']))
    keys = ['ARMA(1,0)','ARMA(2,0)','ARMA(3,0)','ARMA(1,1)','ARMA(2,1)','ARMA(1,2)','ARMA(2,2)',
            'ARIMA(1,1,1)','ARIMA(2,1,1)','ARIMA(2,1,2)']
    values = [RMSE_prices_arma10,RMSE_prices_arma20,RMSE_prices_arma30,RMSE_prices_arma11,
              RMSE_prices_arma21,RMSE_prices_arma12,RMSE_prices_arma22,RMSE_prices_arima111,
              RMSE_prices_arima211,RMSE_prices_arima212]
    result = pd.DataFrame(dict(zip(keys,values)),index = ['RMSE'])
    result = result.T
    result['Rank'] = result['RMSE'].rank()
    return result

RMSE_prices = RMSE_prices_1Y()
sheet.range(1,1).value = RMSE_prices
#%%
def RMSE_returns_1Y():
    """
    Backtest results from AR(I)MA models for the last 20 days in order to make a 
    selection of the best model. 
    
    We base the forecast on the last year.
    
    RMSE FOR RETURNS. 
    """
    warnings.filterwarnings("ignore")
    forecasts_arma10 = arima_mdls[0].forecast(20)
    forecasts_arma20 = arima_mdls[1].forecast(20)
    forecasts_arma30 = arima_mdls[2].forecast(20)
    forecasts_arma11 = arima_mdls[3].forecast(20)
    forecasts_arma21 = arima_mdls[4].forecast(20)
    forecasts_arima111 = arima_mdls[5].forecast(20)
    forecasts_arima211 = arima_mdls[6].forecast(20)
    forecasts_arima212 = arima_mdls[7].forecast(20)
    forecasts_arma12 = arima_mdls[8].forecast(20)
    forecasts_arma22 = arima_mdls[9].forecast(20)
    RMSE_returns_arma11 = np.std(np.array(forecasts_arma11)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma12 = np.std(np.array(forecasts_arma12)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma10 = np.std(np.array(forecasts_arma10)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma20 = np.std(np.array(forecasts_arma20)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma30 = np.std(np.array(forecasts_arma30) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma21 = np.std(np.array(forecasts_arma21) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arma22 = np.std(np.array(forecasts_arma22) - np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arima111 = np.std(np.array(forecasts_arima111)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arima211 = np.std(np.array(forecasts_arima211)-np.array(FTSE_data['%Chg'][-20:]))
    RMSE_returns_arima212 = np.std(np.array(forecasts_arima212)-np.array(FTSE_data['%Chg'][-20:]))
    keys = ['ARMA(1,0)','ARMA(2,0)','ARMA(3,0)','ARMA(1,1)','ARMA(2,1)','ARMA(1,2)','ARMA(2,2)','ARIMA(1,1,1)',
            'ARIMA(2,1,1)','ARIMA(2,1,2)']
    values = [RMSE_returns_arma10,RMSE_returns_arma20,RMSE_returns_arma30,RMSE_returns_arma11,
              RMSE_returns_arma21,RMSE_returns_arma12,RMSE_returns_arma22,
              RMSE_returns_arima111,RMSE_returns_arima211,RMSE_returns_arima212]
    result = pd.DataFrame(dict(zip(keys,values)),index = ['RMSE'])
    result = result.T
    result['Rank RMSE'] = result['RMSE'].rank()
    return result,pd.DataFrame(forecasts_arma11)

def forecast_5days_arma11():
    """
    Next 5 days return for the forecasted return.
    """
    for_arma11 = arima_mdls[3].forecast(25)
    return for_arma11

result_RMSE_returns,forecasts_arma11 = RMSE_returns_1Y()

#%%
forecasts_arma11['Actual returns'] = FTSE_data.iloc[-20:]['%Chg']
forecasts_arma11['Errors'] = forecasts_arma11['predicted_mean']-forecasts_arma11['Actual returns']
for_arma11 = forecast_5days_arma11().iloc[-5:]
sheet.range("E1").value = result_RMSE_returns
sheet.range("E13").value = forecasts_arma11
sheet2 = wb.sheets("Forecasts_ARMA11")
sheet2.range(1,1).value = for_arma11
#%%
"""
Find AR(1) model for rolling data of size 250. The AutoRegressive parameters are retrieved.
AR_params is a list of betas where X(t) = alpha + beta*X(t-1)+eps(t) 
Find AR(2) model for rolling data of size 250.  The AutoRegressive parameters of order 1,2 are retrieved.
"""
ar_params = []
for i in range(50):
    ar_model = ARIMA(np.array(FTSE_data['%Chg'][-300+i:-50+i]),order = (1,0,0)).fit()
    ar_params.append(ar_model.params[1])
    

ar20_params = []
for i in range(50):
    ar20_model = ARIMA(np.array(FTSE_data['%Chg'][-300+i:-50+i]),order = (2,0,0)).fit()
    ar20_params.append(ar20_model.params[[1,2]])


FTSE_main['%chgs'] = FTSE_main['Close'].diff()/FTSE_main['Close']
#%%
def test_dickey_fuller():
    """
    I. 
    Use the ACF value to determine q from ARMA(p,q) model
    Use PACF value to determine p from ARMA(p,q) model.
    Only PACF(1) and ACF(1) have absolute value greater than  10%. 
    Since the ADF = -13.86 < critical value of 1% then the hypothesis of having a unit root is rejected.
    => The stationarity hypothesis is accepted => there is no need for integration => Use ARMA(1,1) model
    II.
    Autocorrelation functions on the entire history of % changes: acf 
    Partial autocorrelation funtions on the entire history of % changes: pacf
    The orders for acf, pacf are with 10 lags.  I start with the second position ([1:])
    because the first value is NaN. 

    Augmented Dickey Fuller is used to check whether there is stationarity. 
    If the test result is below the 99% critical value, we reject the null hypothesis that 
    the process has 1 unit root. 
    """
    acf = statsmodels.tsa.stattools.acf(FTSE_data['%Chg'][1:],10)
    pacf = statsmodels.tsa.stattools.pacf(FTSE_data['%Chg'][1:],10)
    adf = statsmodels.tsa.stattools.adfuller(FTSE_data['%Chg'][1:])
    statsmodels.graphics.tsaplots.plot_pacf(FTSE_data['%Chg'][1:],title = 'PACF for FTSE returns')
    statsmodels.graphics.tsaplots.plot_acf(FTSE_data['%Chg'][1:],title = 'ACF for FTSE returns')
    return acf,pacf,adf

acf,pacf,adf = test_dickey_fuller()
#%%
"""
Compute the RMSE for an AutoRegressive model with exogenous. 
"""
mdl_FTSE_ESG = ARIMA(FTSE_data['%Chg'].iloc[-270:-20],exog = FTSE_data['FTSE returns'].iloc[-270:-20],
            order = (1,0,0)).fit()
mdl_FTSE= ARIMA(FTSE_data['FTSE returns'].iloc[-270:-20],order = (1,0,1)).fit()

forecasts_FTSE = mdl_FTSE.forecast(20)
forecasts_FTSE_ESG = mdl_FTSE_ESG.forecast(20,exog = forecasts_FTSE)
RMSE_ARX = np.std(np.array(forecasts_FTSE_ESG) - np.array(FTSE_data['%Chg'][-20:]))
RMSE_ARX2 = np.linalg.norm(np.array(forecasts_FTSE_ESG) - np.array(FTSE_data['%Chg'][-20:]),2)/np.sqrt(20)
#%%

def store_models_data():
    """
    Store variables arima_mdls and arima_mdls_alltime inside a PICKLE called f using dump method. 
    
    Load the variables using load method. 
    """
    f = open('store.pckl','wb')
    pickle.dump(arima_mdls,f)
    pickle.dump(arima_mdls_alltime,f)
    f.close()
    f = open('store.pckl','rb')
    obj = pickle.load(f)
    f.close()
    #print(obj[0])
store_models_data()



#%%
import sklearn.linear_model as lm

def regression_results():
    """
    Linear Regression:
        y = a*X + b where X = FTSE Main returns and y = FTSE ESG returns. 
        
    Methodology:
        1. I make a linear regression between FTSE returns up to 1 month ago. 
        
        2. I make a forecast of FTSE main index returns
        
        3. I use the coefficients from 1 and the forecasted X using ARMA11, 
        in order to forecast y = FTSE ESG. 
    """
    X = np.array(FTSE_data['FTSE returns'][1:-20],ndmin = 2).T
    y = np.array(FTSE_data['%Chg'][1:-20],ndmin = 2).T
    reg = lm.LinearRegression().fit(X,y)
    coeffs_reg = [reg.coef_,reg.intercept_]
    X_valid = np.array(FTSE_data['FTSE returns'][-20:],ndmin = 2).T
    y_est = reg.predict(X_valid)[:,0]
    errs_reg = FTSE_data['%Chg'][-20:]-y_est
    arma11_main = ARIMA(FTSE_data['FTSE returns'][1:-20],order = (1,0,1)).fit()
    arma11_forecast = arma11_main.forecast(25)
    return errs_reg,arma11_forecast

errs_arma11_reg,forecast_arma11 = regression_results()

sheet3 = wb.sheets("Regression")
sheet3.range(1,1).value = forecast_arma11.iloc[-5:]





