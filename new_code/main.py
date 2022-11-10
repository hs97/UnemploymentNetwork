import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from functions.helpers_labor_market import gen_curlyF_CD, gen_elasticity_Qtheta_CD, gen_tau_CD, r_calib
from functions.cobb_douglas_network import cobb_douglas_network, bar_plot
import pandas as pd

def plot_sector(model, benchmark, var_name, model_name, sectors):
    print(model)
    df = pd.concat([pd.DataFrame(model, index=sectors), pd.DataFrame(benchmark, index=sectors)], axis=1)
    df.columns = [model_name, "benchmark"]
    df.plot(kind="bar")
    plt.ylabel(var_name)
    plt.xlabel("sector")
    plt.show()
    return 

if __name__ == "__main__":
    ##### loading data #####
    data_dir = 'data/clean/'
    dfA      = pd.read_csv(data_dir + 'A.csv')
    dfNames  = pd.read_csv(data_dir + 'sector_names.csv')
    dfParam  = pd.read_csv(data_dir + 'params.csv')
    dfLshare = pd.read_csv(data_dir + 'labor_share.csv')
    dfLabor_market_monthly = pd.read_csv(data_dir + 'labor_market_monthly.csv')
    dfLabor_market_yearly  = pd.read_csv(data_dir + 'labor_market_yearly.csv')
    dfLabor_market_yearly.year  = pd.to_datetime(dfLabor_market_yearly.year, format='%Y')
    dfLabor_market_monthly.date = pd.to_datetime(dfLabor_market_monthly.date)
    dfLabor_market_yearly = dfLabor_market_yearly.rename(columns={'year':'date'})
    dfLabor_market_monthly = dfLabor_market_monthly.sort_values(by=['date', 'BEA_sector'])
    dfLabor_market_yearly  = dfLabor_market_yearly.sort_values(by=['date', 'BEA_sector'])
    dfLabor_market_monthly = dfLabor_market_monthly.dropna(axis=0)
    dfLabor_market_yearly  = dfLabor_market_yearly.dropna(axis=0)
    sectors = dfLshare['BEA_sector']

    ##### reformatting parameters #####
    Omega = np.array(dfA.iloc[:, 1:], dtype='float64')
    phi = np.array(dfParam.φ).reshape(Omega.shape[0],1)
    λ = np.array(dfParam.λ).reshape(Omega.shape[0],1)
    elasticity_fN = np.array(dfParam.α).reshape(Omega.shape[0], 1)
    elasticity_Dc = np.array(dfParam.θ_alt).reshape(Omega.shape[0], 1)
    elasticity_Dc = elasticity_Dc /np.sum(elasticity_Dc)
    η = 0.5
    eta = np.ones_like(elasticity_fN) * η 
    s = 0.03 * np.ones_like(elasticity_fN) # exogenous separation rate (currently set to arbitrary value)
    r = 0.1 * np.ones_like(elasticity_fN) # recruiting cost in relative wage units (currently set to arbitrary value)

    #### Deriving required labor market measures #####
    Nsectors = Omega.shape[0]
    H = np.array(dfLabor_market_yearly.u[-Nsectors-1:-1] + dfLabor_market_yearly.e[-Nsectors-1:-1]).reshape(Nsectors,1)
    L = np.array(dfLabor_market_yearly.e[-Nsectors-1:-1]).reshape(Nsectors,1)
    theta = np.array(dfLabor_market_yearly.v[-Nsectors-1:-1]/dfLabor_market_yearly.u[-Nsectors-1:-1]).reshape(Nsectors,1)
    U = H - L

    elasticity_Qtheta = gen_elasticity_Qtheta_CD(theta,eta)
    curlyF = gen_curlyF_CD(theta,eta,phi,s)
    
    r = r_calib(theta,eta,phi,s,L,0.032)
    tau = gen_tau_CD(theta,eta,phi,s,r*np.ones_like(theta))

    #### Setting up production network ####
    ## Rigid nominal wages ##
    elasticity_wtheta = np.zeros_like(Omega)
    elasticity_wA = np.zeros_like(Omega)
    cobb_douglas_rigid_nominal = cobb_douglas_network(Omega, elasticity_Dc, elasticity_fN, elasticity_Qtheta, tau, curlyF, elasticity_wtheta, elasticity_wA)

    # shock hyperparameters
    shock_mag = 0.01
    shock_sign = 1
    shock_sec = 3 # which sector to shock

    # absent shocks
    dlogA_null = np.zeros_like(elasticity_Dc)
    dlogH_null = np.zeros_like(elasticity_Dc)

    # shocks to all sectors
    dlogA_all = shock_mag * shock_sign * np.ones_like(dlogA_null)
    dlogH_all = shock_mag * shock_sign * np.ones_like(dlogH_null)

    # sector-specific shocks
    dlogA_sec = np.zeros_like(elasticity_Dc)
    dlogH_sec = np.zeros_like(elasticity_Dc)
    dlogA_sec[shock_sec] = shock_mag * shock_sign
    dlogH_sec[shock_sec] = shock_mag * shock_sign

    # Shocks to economies with rigid wages
    tech_shock_nominal = cobb_douglas_rigid_nominal.shocks(dlogA_all, dlogH_null, H, L, U)
    tech_shock_nominal_sec = cobb_douglas_rigid_nominal.shocks(dlogA_sec, dlogH_null, H, L, U)
    H_shock_nominal = cobb_douglas_rigid_nominal.shocks(dlogA_null, dlogH_all, H, L, U)
    H_shock_nominal_sec = cobb_douglas_rigid_nominal.shocks(dlogA_null, dlogH_sec, H, L, U)

    ## Wages move with aggregate price level ##
    gamma = 0.25
    I = np.eye(Omega.shape[0])
    term1 = I - (gamma*np.ones_like(elasticity_Dc)) @ elasticity_Dc.T @ cobb_douglas_rigid_nominal.Psi @ np.diag(elasticity_fN.flatten())
    Inv_term = np.linalg.inv(term1)
    elasticity_wA = -Inv_term @ (gamma * np.ones_like(elasticity_Dc)) @ elasticity_Dc.T @ cobb_douglas_rigid_nominal.Psi 
    elasticity_wtheta = elasticity_wA @ np.diag(elasticity_fN.flatten()) @ np.diag(tau.flatten()) @ np.diag(elasticity_Qtheta.flatten())
    cobb_douglas_rigid_real = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
    tech_shock_real = cobb_douglas_rigid_real.shocks(dlogA_all, dlogH_null, H, L, U) 
    tech_shock_real_sec = cobb_douglas_rigid_real.shocks(dlogA_sec, dlogH_null, H, L, U) 
    H_shock_real = cobb_douglas_rigid_real.shocks(dlogA_null, dlogH_all, H, L, U)

    ## Wages move with sectoral price level ##
    I = np.eye(Omega.shape[0])
    term1_sectoral = I - (gamma*cobb_douglas_rigid_nominal.Psi) @ np.diag(elasticity_fN.flatten())
    Inv_term = np.linalg.inv(term1_sectoral)
    elasticity_wA = -Inv_term @ (gamma*cobb_douglas_rigid_nominal.Psi)
    elasticity_wtheta = elasticity_wA @ np.diag(elasticity_fN.flatten()) @ np.diag(tau.flatten()) @ np.diag(elasticity_Qtheta.flatten())
    cobb_douglas_sectoral_real = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
    tech_shock_sectoral = cobb_douglas_sectoral_real.shocks(dlogA_all, dlogH_null, H,L,U) 
    tech_shock_sectoral_sec = cobb_douglas_sectoral_real.shocks(dlogA_sec, dlogH_null, H, L, U) 
    H_shock_sectoral = cobb_douglas_sectoral_real.shocks(dlogA_null, dlogH_all, H, L, U)
    H_shock_sectoral_sec = cobb_douglas_sectoral_real.shocks(dlogA_null, dlogH_sec, H, L, U)


    ## Wages rise one for one with own sector A, do no change with tightness ##
    elasticity_wA = np.eye(Omega.shape[0])
    elasticity_wtheta = np.zeros_like(Omega)
    cobb_douglas_eyeA = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
    tech_shock_eyeA = cobb_douglas_eyeA.shocks(dlogA_all, dlogH_null, H, L, U) 
    H_shock_eyeA = cobb_douglas_eyeA.shocks(dlogA_null, dlogH_all, H, L, U)

    '''
    #### Figures ####
    plot_sector(tech_shock_real.dlogy, tech_shock_nominal.dlogy, var_name='dlogy', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)
    plot_sector(tech_shock_real.dUrate, tech_shock_nominal.dUrate, var_name='change in u', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)
    plot_sector(tech_shock_real.dlogtheta, tech_shock_nominal.dlogtheta, var_name=r'd$\log\theta$', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)

    # Impact of sector level shocks
    plot_sector(tech_shock_real_sec.dlogy, tech_shock_nominal_sec.dlogy, var_name='dlogy', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)
    plot_sector(tech_shock_real_sec.dUrate, tech_shock_nominal_sec.dUrate, var_name='change in u', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)
    plot_sector(tech_shock_real_sec.dlogtheta, tech_shock_nominal_sec.dlogtheta, var_name=r'd$\log\theta$', model_name='partially rigid wages', sectors=range(Nsectors)) #sectors)
    '''


    #Impact tech shocks, agg included
    networks = [tech_shock_nominal, tech_shock_sectoral]
    sector_names = list(dfNames.BEA_sector_short) + list(['Agg.'])
    varname = 'dlogy'
    aggname = 'dlogY'
    title   = 'Response to 1% productivity shock in all sectors' 
    xlab    = ''
    ylab    = 'Log change in real output'
    labels  = ['Constant wage', 'Partial adjustment']
    save_path = 'output/tech_shock_fixed_sectoral_output_all.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)

    #unemployment rate changes
    varname = 'dUrate'
    aggname = 'dUrate_agg'
    ylab    = ''
    save_path = 'output/tech_shock_fixed_sectoral_Urate_all.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)

    #Sectoral shocks
    networks = [tech_shock_nominal_sec, tech_shock_sectoral_sec]
    varname = 'dlogy'
    aggname = 'dlogY'
    title   = 'Response to 1% productivity shock in durables' 
    xlab    = ''
    ylab    = 'Log change in real output'
    labels  = ['Constant wage', 'Partial adjustment']
    save_path = 'output/tech_shock_fixed_sectoral_output_durables.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)
    
    #unemployment rate changes
    varname = 'dUrate'
    aggname = 'dUrate_agg'
    ylab    = ''
    save_path = 'output/tech_shock_fixed_sectoral_Urate_durables.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)

    #impact of H shock
    networks = [H_shock_nominal, H_shock_sectoral]
    sector_names = list(dfNames.BEA_sector_short) + list(['Agg.'])
    varname = 'dlogy'
    aggname = 'dlogY'
    title   = 'Response to 1% labor force shock in all sectors' 
    xlab    = ''
    ylab    = 'Log change in real output'
    labels  = ['Constant wage', 'Partial adjustment']
    save_path = 'output/H_shock_fixed_sectoral_output_all.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)

    #unemployment rate changes
    varname = 'dUrate'
    aggname = 'dUrate_agg'
    ylab    = ''
    save_path = 'output/H_shock_fixed_sectoral_Urate_all.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)
    
    #Sectoral shocks
    networks = [H_shock_nominal_sec, H_shock_sectoral_sec]
    varname = 'dlogy'
    aggname = 'dlogY'
    title   = 'Response to 1% labor force shock in durables' 
    xlab    = ''
    ylab    = 'Log change in real output'
    labels  = ['Constant wage', 'Partial adjustment']
    save_path = 'output/H_shock_fixed_sectoral_output_durables.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)
    
    #unemployment rate changes
    varname = 'dUrate'
    aggname = 'dUrate_agg'
    ylab    = ''
    save_path = 'output/H_shock_fixed_sectoral_Urate_durables.png'
    bar_plot(networks, sector_names, varname, aggname, title, xlab, ylab, labels, save_path, rotation=30, fontsize=15, barWidth = 0.25, dpi=300)

    print('done')
