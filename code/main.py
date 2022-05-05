import numpy as np
import matplotlib.pyplot as plt
from functions.optimal_unemployment import m_cd, mu_cd, Lstar, Lones, objective, ustar, Mindex, mismatch_estimation
import pandas as pd

# loading data
data_dir = 'data/clean/'
dfA      = pd.read_csv(data_dir+'A.csv')
dfParam  = pd.read_csv(data_dir+'params.csv')
dfLshare = pd.read_csv(data_dir+'labor_share.csv')
dfLabor_market_monthly= pd.read_csv(data_dir+'labor_market_monthly.csv')
dfLabor_market_yearly= pd.read_csv(data_dir+'labor_market_yearly.csv')
dfLabor_market_yearly.year = pd.to_datetime(dfLabor_market_yearly.year,format='%Y')
dfLabor_market_monthly.date = pd.to_datetime(dfLabor_market_monthly.date)
dfLabor_market_yearly = dfLabor_market_yearly.rename(columns={'year':'date'})

dfLabor_market_monthly = dfLabor_market_monthly.sort_values(by=['date','BEA_sector'])
dfLabor_market_yearly  = dfLabor_market_yearly.sort_values(by=['date','BEA_sector'])
dfLabor_market_monthly = dfLabor_market_monthly.dropna(axis=0)
dfLabor_market_yearly = dfLabor_market_yearly.dropna(axis=0)


# reformatting parameters
A = np.array(dfA.iloc[:,1:],dtype='float64')
φ = np.array(dfParam.φ)
λ = np.array(dfParam.λ)
α = np.array(dfParam.α)
θ = np.array(dfParam.θ)
η = 0.5

# Sahin et al baseline
sahin_yearly = mismatch_estimation(dfLabor_market_yearly,objective,φ,η,np.ones_like(φ),np.ones_like(φ),m_cd,mu_cd,Lones,guessrange=0.01,ntrue=10,tol=1e-8)
#sahin_monthly = mismatch_estimation(dfLabor_market_monthly,objective,φ,η,np.ones_like(φ),np.ones_like(φ),m_cd,mu_cd,Lones,guessrange=0.01,ntrue=10,tol=1e-8)
#sahin_monthly.mHP(10)

# With production network
networks_yearly = mismatch_estimation(dfLabor_market_yearly,objective,φ,η,λ,α,m_cd,mu_cd,Lstar,guessrange=0.01,ntrue=10,tol=1e-8)
#networks_monthly = mismatch_estimation(dfLabor_market_monthly,objective,φ,η,λ,α,m_cd,mu_cd,Lstar,guessrange=0.01,ntrue=10,tol=1e-8)
#networks_monthly.mHP(10)

plt.plot(networks_yearly.output.index, networks_yearly.output['mismatch_index'])
plt.plot(sahin_yearly.output.index, sahin_yearly.output['mismatch_index'])
plt.show()


print('done')