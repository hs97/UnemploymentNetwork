import numpy as np
import matplotlib.pyplot as plt
from functions.optimal_unemployment import m_cd, mu_cd, Lstar, Lones, objective, ustar, Mindex
import pandas as pd

data_dir = 'data/clean/'
dfA      = pd.read_csv(data_dir+'A.csv')
dfParam  = pd.read_csv(data_dir+'params.csv')
dfLshare = pd.read_csv(data_dir+'labor_share.csv')
dfJOLTs_monthly = pd.read_csv(data_dir+'JOLTs_monthly.csv')
dfJOLTs_yearly  = pd.read_csv(data_dir+'JOLTs_yearly.csv')

A = np.array(dfA.iloc[:,1:],dtype='float64')
φ = np.array(dfParam.φ)
λ = np.array(dfParam.λ)
α = np.array(dfParam.α)
θ = np.array(dfParam.θ)
η = 0.5

# Sahin et al baseline
vraw = np.array(dfJOLTs_yearly.v[dfJOLTs_yearly.year==2001])
uraw = np.array(dfJOLTs_yearly.v[dfJOLTs_yearly.year==2001])
ustar(objective,)

# With production network

print('done')