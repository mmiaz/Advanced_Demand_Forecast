# import packages
import os
import pandas as pd
import numpy as np
from helpers import *


def main():
    # Test Inupt Template
    tsquantity_origin = pd.read_csv("ADF_test.csv").iloc[:,0:2]

    # Format Transformation
    tsquantity_origin['ds'] = pd.to_datetime(tsquantity_origin['ds'])
    tsquantity = tsquantity_origin.set_index('ds')
    tsquantity_prophet, tsquantity_lr = tsquantity_origin.copy(), tsquantity_origin.copy()
    tsquantity_lr['ds'] = tsquantity_lr['ds'].astype(np.int64)//10**9

    # check whether the last 60% are 0s
    if False in list(tsquantity_origin['y'][int(0.4*len(tsquantity)):] == np.zeros(len(tsquantity_origin['y'][int(0.4*len(tsquantity)):]))):
        print('please continue the below analysis')
    else:
        print('future prediction are all zeros')


    # modeling
    # 1. characteristics checks
    tsquantity = transform_zeors(tsquantity)
    stationary_diff = test_stationarity(tsquantity)
    test_seasonality(tsquantity)

    # 2. model fitting
    if len(tsquantity_origin)<36:
        result = setA(tsquantity)
    else:
        result, model_autoarima, model_sautoarima, model_prophet, model_lr, model_dt = setA_B(tsquantity,stationary_diff,tsquantity_prophet,tsquantity_lr)
        
        # pair the models with column name for future usage
        models = dict.fromkeys(list(result.columns)[5:])
        models['setB_autoarima'] = model_autoarima
        models['setB_sautoarima'] = model_sautoarima
        models['setB_markovreg'] = model_markovreg
        models['setB_prophet'] = model_prophet
        models['setB_lr'] = model_lr
        models['setB_dt'] = model_dt

    # 3. get result
    result_rmse,result_select = findbest(result)



if __name__ == "__main__":
   main()