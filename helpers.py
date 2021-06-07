# import packages
import os
import pandas as pd
import numpy as np
import warnings
import scipy
from scipy import stats
import timeit

# Outlier detection
from sklearn.covariance import EllipticEnvelope

# Stationarity test & seasonality
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Forceasting with decompasable model / Traditional Time Series Model
from pmdarima import auto_arima
from fbprophet import Prophet

# For marchine Learning Approach
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.linear_model import LinearRegression#, RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from math import sqrt
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')



def transform_zeors(tsquantity):
    """ outlier detection -- Anomaly Detection using Gaussian Distribution

    Parameters
    -------------
    tsquantity: dataframe
        time series with date and value

    Returns
    -------------
    tsquantity: dataframe
        if 0 in outliers, then use mean for (x-1) and (x+1), else ignore the outliers
    """
    if tsquantity[tsquantity['y']==0].empty == False:
        outliers_fraction = round(len(tsquantity[tsquantity['y']==0])/len(tsquantity),2)
        try:
            envelope = EllipticEnvelope(contamination = outliers_fraction)
            X = tsquantity['y'].values.reshape(-1,1)
            envelope.fit(X)
        except ValueError:
            envelope = EllipticEnvelope(contamination = outliers_fraction, support_fraction=1)
            X = tsquantity['y'].values.reshape(-1,1)
            envelope.fit(X)
        X_quantity = pd.DataFrame(tsquantity['y'])
        X_quantity['deviation'] = envelope.decision_function(X)
        X_quantity['anomaly'] = envelope.predict(X)
        if X_quantity[X_quantity['anomaly']==-1][X_quantity['y']==0].empty == True:
            pass
        else:
            indexlist = list(X_quantity[X_quantity['anomaly']==-1][X_quantity['y']==0].index)
            if indexlist[0] == list(tsquantity.index)[0]:
                tsquantity.loc[indexlist[0]] = tsquantity.loc[indexlist[1]]/2
                for i in range(1, len(indexlist)-1):
                    tsquantity.loc[indexlist[i]] = (tsquantity.loc[indexlist[i-1]]+tsquantity.loc[indexlist[i+1]])/2
            else:
                for i in range(len(indexlist)-1):
                    tsquantity.loc[indexlist[i]] = (tsquantity.loc[indexlist[i-1]]+tsquantity.loc[indexlist[i+1]])/2
            tsquantity.loc[indexlist[-1]] = tsquantity.loc[indexlist[-2]]/2

    return tsquantity


def test_stationarity(timeseries):
    """ Test time series stationarity

    Parameters
    -------------
    timeseries: dataframe 
        time series with date and value

    Returns
    -------------
    d: int
        Find the correct d for the stationary time series
    """
    ADFtest_pvalue = adfuller(timeseries['y'],autolag='AIC')[1]
    if ADFtest_pvalue <= 0.05:
        print('The Time Series is stationary: p value = %a'%ADFtest_pvalue)
        d = 0
    else:
        for i in [1,2]:
            new = timeseries['y'] - timeseries['y'].shift(i)
            ADFtest_pvalue = adfuller(new.dropna(),autolag='AIC')[1]
            if ADFtest_pvalue <= 0.05:
                print('When d=%i, time series is then stationary'%i)
                d = i
                break
        if ADFtest_pvalue > 0.05:
            print('manually check')
    return d


def test_seasonality(tsquantity):
    """ Helper function to visualize the time series for seasonality checking

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    None, plot
    """
    decomposition = seasonal_decompose(tsquantity, model='additive')
    seasonality = decomposition.seasonal
    plt.figure(figsize=(11,6))
    plt.plot(seasonality)



# Two set of time series modelings --- set A for input with not enough data and vice versa for set B
# Our goal is to predict future 6 periods' values
# set A
def setA_mean(tsquantity):
    """ Use mean value of recent 6 data points for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    None, applies changes to tsquantity directly
    """
    tsquantity['setA_mean'] = tsquantity['y']
    for i in range(6):
        tsquantity['setA_mean'][-6+i] = int(np.mean(tsquantity['setA_mean'][:-6+i]))


def setA_rwalk(tsquantity):
    """ Use random walk methodology for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    None, applies changes to tsquantity directly
    """
    tsquantity['setA_rwalk'] = tsquantity['y']
    for i in range(6):
        tsquantity['setA_rwalk'][-6+i] = tsquantity['setA_rwalk'][-7+i]


def setA_rwalk_seasonal(tsquantity):
    """ Use seasonal random walk methodology for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    None, applies changes to tsquantity directly
    """
    if len(tsquantity) >= 18:
        tsquantity['setA_rwalk_seasonal'] = tsquantity['y']
        for i in range(6):
            tsquantity['setA_rwalk_seasonal'][-6+i] = tsquantity['setA_rwalk_seasonal'][-18+i]
    else:
        print('The length is less than 18 for seasonal random walk.')


# Set B
def setB_Croston_TSB(tsquantity,extra_periods=1,alpha=0.4,beta=0.4):
    """ Use crostons method for prediction when there are too many 0s in the time series

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    None, applies changes to tsquantity directly
    """
    try:
        d = np.array(tsquantity['y']) # Transform the input into a numpy array
        cols = len(d) # Historical period length
        d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
        
        #level (a), probability(p) and forecast (f)
        a,p,f = np.full((3,cols+extra_periods),np.nan)
        
        # Initialization
        first_occurence = np.argmax(d[:cols]>0)
        a[0] = d[first_occurence]
        p[0] = 1/(1 + first_occurence)
        f[0] = p[0]*a[0]
        
        # Create all the t+1 forecasts
        for t in range(0,cols): 
            if d[t] > 0:
                a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
                p[t+1] = beta*(1) + (1-beta)*p[t]  
            else:
                a[t+1] = a[t]
                p[t+1] = (1-beta)*p[t]       
            f[t+1] = p[t+1]*a[t+1]
            
        # Future Forecast
        f[cols+1:cols+extra_periods] = f[cols]
        tsquantity['setB_Croston_TSB'] = tsquantity['y']
        tsquantity['setB_Croston_TSB'][-6:] = f[-7:-1].astype(int)
        #df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    except:
        tsquantity['setB_Croston_TSB'] = np.nan
    

def setB_autoarima(tsquantity,stationary_diff):
    """ Use autoARIMA for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
    
    stationary_diff: int
        corresponding d for the stationary time series

    Returns
    -------------
    autoarima_model: model 
        autoarima_model for future calls/checks
    """
    tsquantity['setB_autoarima'] = tsquantity['y']

    # train/test split
    train = tsquantity['y'][:-6]

    #building the model
    autoarima_model = auto_arima(train, start_p=1, start_q=1,max_p=4, max_q=4, d=stationary_diff,
                               start_P=0, seasonal=False, trace=True, error_action='ignore',  
                               suppress_warnings=True, stepwise=True, n_jobs=-1)
    autoarima_model.fit(train)
    tsquantity['setB_autoarima'][-6:] = autoarima_model.predict(n_periods=6).astype(int)
    return autoarima_model


def setB_sautoarima(tsquantity,stationary_diff):
    """ Use auto seasonal ARIMA for prediction if Seasonality encountered
    (default: Seasonality = 1 year)

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
    
    stationary_diff: int
        corresponding d for the stationary time series
        
    Returns
    -------------
    autoarima_model: model 
        autoarima_model for future calls/checks
    """
    tsquantity['setB_sautoarima'] = tsquantity['y']
    # train/test split
    train = tsquantity['y'][:-6]

    #building the model
    autoarima_model = auto_arima(train, start_p=1, start_q=1,max_p=4, max_q=4, m=12, d=stationary_diff,
                                 start_P=0, seasonal=True, trace=True, error_action='ignore',  
                                 suppress_warnings=True, stepwise=True, n_jobs=-1)
    #print(autoarima_model.aic())
    autoarima_model.fit(train)
    tsquantity['setB_sautoarima'][-6:] = autoarima_model.predict(n_periods=6).astype(int)
    return autoarima_model


def setB_prophet(tsquantity,tsquantity_prophet):
    """ Use facebook prophet for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
    
    tsquantity_prophet: dataframe
        specific format to prophet input
        
    Returns
    -------------
    model: model 
        prophet model for future calls/checks
    """
    tsquantity['setB_prophet'] = tsquantity['y']
    model = Prophet(changepoint_prior_scale=0.1)
    model = model.fit(tsquantity_prophet[:-6])
    tsquantity['setB_prophet'][-6:] = model.predict(tsquantity_prophet[-6:])['yhat'].astype(int)
    tsquantity['setB_prophet'] = np.where(tsquantity['setB_prophet']<0,0,tsquantity['setB_prophet'])
    return model


def setB_MarkovRegression(tsquantity):
    """ Use Markov switching dynamic regression models regression models for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
        
    Returns
    -------------
    model_fit: model 
        Markov switching dynamic regression model for future calls/checks
    """
    try:
        # model buidling # (a switching mean is the default of the MarkovRegession model)
        model = sm.tsa.MarkovRegression(tsquantity['y'], k_regimes=2)
        model_fit = model.fit()
        tsquantity['setB_markovreg'] = tsquantity['y']
        tsquantity['setB_markovreg'][-6:] = model_fit.predict()[-6:].astype(int)
        return model_fit
    except:
        return None
   


def setB_lr(tsquantity,tsquantity_lr):
    """ Use Simple linear regression models for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
    
    tsquantity_lr: dataframe
        specific format to linear regression input
        
    Returns
    -------------
    model_lr: model 
        regression model for future calls/checks
    """
    tsquantity['setB_lr'] = tsquantity['y']
    model_lr = LinearRegression()
    model_lr.fit(tsquantity_lr['ds'][:-6].values.reshape(-1,1),tsquantity_lr['y'][:-6])
    tsquantity['setB_lr'][-6:] = model_lr.predict(tsquantity_lr['ds'][-6:].values.reshape(-1,1)).astype(int)
    return model_lr


def setB_dt(tsquantity,tsquantity_lr):
    """ Use Decision Tree models for prediction

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value
    
    tsquantity_lr: dataframe
        specific format to linear regression input
        
    Returns
    -------------
    model_lr: model 
        regression model for future calls/checks
    """
    try:
        parameters={'min_samples_split' : range(5,10,1),'max_depth': range(3,8,1)}
        model_tree =DecisionTreeClassifier()
        model= GridSearchCV(model_tree,parameters)
        model.fit(tsquantity_lr['ds'][:-6].values.reshape(-1,1),tsquantity_lr['y'][:-6])
        model_dt = model.best_estimator_
        tsquantity['setB_dt'] = tsquantity['y']
        tsquantity['setB_dt'][-6:] = model_dt.predict(tsquantity_lr['ds'][-6:].values.reshape(-1,1))
        return model_dt
    except:
        return None


def setA(tsquantity):
    """ Call set A baseline models 

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    Returns
    -------------
    tsquantity: dataframe 
        time series with new predicted set of values
    """
    setA_mean(tsquantity)
    setA_rwalk(tsquantity)
    setA_rwalk_seasonal(tsquantity)
    return tsquantity


def setA_B(tsquantity,stationary_diff,tsquantity_prophet,tsquantity_lr):
    """ Call set B advanced models 

    Parameters
    -------------
    tsquantity: dataframe 
        time series with date and value

    stationary_diff: int
        corresponding d for the stationary time series
        
    tsquantity_prophet: dataframe
        specific format to prophet input

    tsquantity_lr: dataframe
        specific format to linear regression input

    Returns
    -------------
    tsquantity: dataframe 
        time series with new predicted set of values
    
    model_autoarima: model
    model_sautoarima: model
    model_prophet: model
    model_lr: model
    model_dt: model
    model_markovreg: model
        a set of fitted models for future calls
    """
    setA_mean(tsquantity)
    setA_rwalk(tsquantity)
    setA_rwalk_seasonal(tsquantity)
    setB_Croston_TSB(tsquantity)
    model_autoarima = setB_autoarima(tsquantity,stationary_diff)
    model_sautoarima = setB_sautoarima(tsquantity,stationary_diff)
    model_prophet = setB_prophet(tsquantity,tsquantity_prophet)
    model_lr = setB_lr(tsquantity,tsquantity_lr)
    model_dt = setB_dt(tsquantity,tsquantity_lr)
    model_markovreg = setB_MarkovRegression(tsquantity)
    tsquantity[tsquantity<0] = 0
    return tsquantity, model_autoarima, model_sautoarima, model_prophet, model_lr, model_dt, model_markovreg


def findbest(result):
    """ use RMSE to find the best predicted model
        and plot the best predictions for validation set

    Parameters
    -------------
    result: dataframe 
        time series results

    Returns
    -------------
    result_rmse: dataframe 
        time series results with corresponding RMSE

    result_select: dataframe 
        clean time series results with selected best values
    """
    # calculate rmse
    colname = list(result.columns)[1:]
    result_rmse = dict.fromkeys(colname)

    for i in colname:
        result_rmse[i] = int(sqrt(mean_squared_error(result['y'][-6:],result[i][-6:])))
    result_rmse = dict(sorted(result_rmse.items(), key = lambda x: x[1]))

    try:
        for i in list(models.keys()):
            if i == None:
                del result_rmse[i]
    except:
        pass

    if list(result_rmse.keys())[0] == 'setA_rwalk' and list(result_rmse.keys())[1] == 'setA_rwalk_seasonal':
        result_select = result[['y',list(result_rmse.keys())[1]]]
    else:
        result_select = result[['y',list(result_rmse.keys())[0]]]
    
    #plot the best predictions for validation set
    plt.figure(figsize=(11,7))
    plt.plot(result_select['y'], label='True', marker='.', color='b')
    plt.plot(result_select[list(result_select.keys())[1]][-6:], label=list(result_select.keys())[1], marker='.', color='r')
    plt.legend(loc='best')
    plt.show()
    return result_rmse,result_select