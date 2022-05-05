import numpy as np
import matplotlib.pyplot as plt
from functions.optimal_unemployment import m_cd, mu_cd, Lstar, Lones, ustar_objective, mismatch_estimation
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
dfLabor_market_yearly  = dfLabor_market_yearly.dropna(axis=0)


# reformatting parameters
A = np.array(dfA.iloc[:,1:],dtype='float64')
φ = np.array(dfParam.φ)
λ = np.array(dfParam.λ)
α = np.array(dfParam.α)
θ = np.array(dfParam.θ)
η = 0.5

param_sahin = {'A':A,'φ':φ,'λ':np.ones_like(λ),'α':np.ones_like(α),'θ':θ,'η':η,'mfunc':m_cd,'mufunc':mu_cd,'Lfunc':Lones,'objective':ustar_objective}

# Sahin et al baseline
sahin_yearly = mismatch_estimation(dfLabor_market_yearly,param_sahin,guessrange=0.01,ntrue=2,tol=1e-8)
sahin_monthly = mismatch_estimation(dfLabor_market_monthly,param_sahin,guessrange=0.01,ntrue=2,tol=1e-8)
sahin_monthly.mHP(10,'sahin_monthly',600)
sahin_monthly.sector_level('sahin_monthly',600)

# With production network
param_networks = {'A':A,'φ':φ,'λ':λ,'α':α,'θ':θ,'η':η,'mfunc':m_cd,'mufunc':mu_cd,'Lfunc':Lstar,'objective':ustar_objective}
networks_yearly = mismatch_estimation(dfLabor_market_yearly,param_networks,guessrange=0.01,ntrue=2,tol=1e-8)
networks_monthly = mismatch_estimation(dfLabor_market_monthly,param_networks,guessrange=0.01,ntrue=2,tol=1e-8)
networks_monthly.mHP(10,'networks_monthly',600)
networks_monthly.sector_level('networks_monthly',600)

# Constant λα, different e across industries
param_cλα = {'A':A,'φ':φ,'λ':np.ones_like(λ),'α':np.ones_like(α),'θ':θ,'η':η,'mfunc':m_cd,'mufunc':mu_cd,'Lfunc':Lstar,'objective':ustar_objective}
constantλα_monthly = mismatch_estimation(dfLabor_market_monthly,param_cλα,guessrange=0.01,ntrue=2,tol=1e-8)
constantλα_monthly.mHP(10,'constantλα_monthly',600)

# Constant e, different λα across industries
param_ce = {'A':A,'φ':φ,'λ':λ,'α':α,'θ':θ,'η':η,'mfunc':m_cd,'mufunc':mu_cd,'Lfunc':Lstar,'objective':ustar_objective}
dfLabor_market_monthly_constant_e   = dfLabor_market_monthly.copy(deep=True)
dfLabor_market_monthly_constant_e.e = dfLabor_market_monthly.e.mean() 
constante_monthly = mismatch_estimation(dfLabor_market_monthly_constant_e,param_cλα,guessrange=0.01,ntrue=2,tol=1e-8)
constante_monthly.mHP(10,'constante_monthly',600)

# Figures 
fig_aggregate_mismatch, ax = plt.subplots(1,1,dpi=600)
plt.rcParams['font.size'] = '8'
ax.plot(sahin_monthly.output.index,sahin_monthly.output.mismatch_trend,'-k',label='Horizontal Economy')
ax.plot(networks_monthly.output.index,networks_monthly.output.mismatch_trend,'--r',label='Network Economy')
ax.set_xlabel('Date')
ax.set_ylabel('Mismatch Index')
ax.legend()
plt.savefig('code/output/mismatch_index_comparison.png')

fig_aggregate_mismatch_2ax, ax1 = plt.subplots(1,1,dpi=600)
plt.rcParams['font.size'] = '8'
ax1.plot(sahin_monthly.output.index,sahin_monthly.output.mismatch_trend,'-k',label='Horizontal Economy')
ax1.set_xlabel('Date')
ax1.set_ylabel('Horizontal Economy Mismatch Index')
ax2 = ax1.twinx()
ax2.plot(networks_monthly.output.index,networks_monthly.output.mismatch_trend,'--r',label='Network Economy')
ax2.set_xlabel('Date')
ax2.set_ylabel('Network Economy Mismatch Index')
ax1.legend()
ax2.legend()
plt.savefig('code/output/mismatch_index_comparison_2ax.png')

fig_network_mismatch_decomp, ax = plt.subplots(1,1,dpi=600)
plt.rcParams['font.size'] = '8'
ax.plot(networks_monthly.output.index,networks_monthly.output.mismatch_trend,'-k',label='Network Economy')
ax.plot(constantλα_monthly.output.index,constantλα_monthly.output.mismatch_trend,'--r',label='Constant λα Across Sectors')
ax.plot(constante_monthly.output.index,constante_monthly.output.mismatch_trend,'--b',label='Constant Existing Employment Across Sectors')
ax.set_xlabel('Date')
ax.set_ylabel('Mismatch Index')
ax.legend()
plt.savefig('code/output/network_mismatch_decomp.png')

fig_aggregate_mismatch_constante, ax = plt.subplots(1,1,dpi=600)
plt.rcParams['font.size'] = '8'
ax.plot(sahin_monthly.output.index,sahin_monthly.output.mismatch_trend,'-k',label='Horizontal Economy')
ax.plot(constante_monthly.output.index,constante_monthly.output.mismatch_trend,'--r',label='Constant Existing Workers Network Economy')
ax.set_xlabel('Date')
ax.set_ylabel('Mismatch Index')
ax.legend()
plt.savefig('code/output/mismatch_index_comparison_constante.png')


print('done')