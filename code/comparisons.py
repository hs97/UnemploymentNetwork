import numpy as np
import matplotlib.pyplot as plt
from functions.optimal_unemployment import m_cd, mu_cd, Lstar, Lones, ustar_objective, mismatch_estimation
import pandas as pd

if __name__ == "__main__":
    # This section compares mismatch indices under different specifications
    # Specifically, we are interested in what happens when the Sahin et al benchmark has diminishing return to scale of labor
    # and when our production network case has constant return to scale of labor. 
    # loading data
    data_dir = 'data/clean/'
    dfA      = pd.read_csv(data_dir + 'A.csv')
    dfParam  = pd.read_csv(data_dir + 'params.csv')
    dfLshare = pd.read_csv(data_dir + 'labor_share.csv')
    dfLabor_market_monthly= pd.read_csv(data_dir + 'labor_market_monthly.csv')
    dfLabor_market_monthly.date = pd.to_datetime(dfLabor_market_monthly.date)
    dfLabor_market_monthly = dfLabor_market_monthly.sort_values(by=['date', 'BEA_sector'])
    dfLabor_market_monthly = dfLabor_market_monthly.dropna(axis=0)

    # reformatting parameters
    A = np.array(dfA.iloc[:, 1:], dtype='float64')
    φ = np.array(dfParam.φ)
    λ = np.array(dfParam.λ)
    λ_alt = np.array(dfParam.λ_alt)

    α = np.array(dfParam.α)
    θ = np.array(dfParam.θ)
    θ_alt = np.array(dfParam.θ)
    γ = np.array(dfParam.γ)

    η = 0.5
    # This solves the objective function that equalizes
    # phi_i m_u
    # for all sectors
    param_sahin = {'A': A, 'φ': φ, 'λ': np.ones_like(λ), 'α': np.ones_like(α), 'θ': θ, 'η': η, 'mfunc': m_cd,
                   'mufunc': mu_cd, 'Lfunc': Lones, 'objective': ustar_objective}
    #sahin_monthly = mismatch_estimation(dfLabor_market_monthly,param_sahin, guessrange=0.01, ntrue=2, tol=1e-8)
    # sahin_monthly.mHP(10, 'sahin_monthly_CRS', 600)

    # With production network
    # This solves the objective function that equalizes
    # \lambda_i \phi_i m_u
    # for all sectors
    # This comes from assuming idiosyncratic productivity for firms at each time,
    # such that the marginal product of labor in each sector equalizes
    param_networks = {'A': A, 'φ': φ, 'λ': λ, 'α': np.ones_like(α), 'θ': θ, 'η': η, 'mfunc': m_cd,
                      'mufunc': mu_cd, 'Lfunc': Lones, 'objective': ustar_objective}
    networks_monthly = mismatch_estimation(dfLabor_market_monthly, param_networks, guessrange=0.01, ntrue=2, tol=1e-8)
    networks_monthly.mHP(10, 'networks_monthly_CRS', 600)

    # This solves the objective function that equalizes
    # \theta_i m_u_i / L^*_i
    # for all sectors
    '''
    param_sahin_DRS = {'A': A, 'φ': φ, 'λ': θ_alt, 'α': np.ones_like(α), 'θ': θ, 'η': η, 'mfunc': m_cd,
                       'mufunc': mu_cd, 'Lfunc': Lstar, 'objective': ustar_objective}
    sahin_DRS_monthly = mismatch_estimation(dfLabor_market_monthly, param_sahin_DRS, guessrange=0.05, ntrue=1, tol=1e-8)
    sahin_DRS_monthly.mHP(10, 'sahin_monthly_DRS', 600)
    '''
    # With production network
    # This solves the objective function that equalizes
    # \lambda_i \alpoha_i m_u_i / L^*_i
    # for all sectors
    param_networks = {'A': A, 'φ': φ, 'λ': λ_alt, 'α': α, 'θ': θ, 'η': η, 'mfunc': m_cd,
                      'mufunc': mu_cd, 'Lfunc': Lstar, 'objective': ustar_objective}
    #networks_monthly = mismatch_estimation(dfLabor_market_monthly, param_networks, guessrange=0.01, ntrue=2, tol=1e-8)
    #networks_monthly.mHP(10, 'networks_monthly_DRS', 600)

