import numpy as np
import pandas as pd
import functions.multi_occupation_network as multi_occupation_network
import functions.nominal_rigidity_network as nominal_rigidity_network
import matplotlib.pyplot as plt
from  functions.network_figures import bar_plot

#### Reading in calibration params ####
do_unemployment = 1
A_spec = 1
data_dir = '../data/clean/'
dfTau    = pd.read_csv(data_dir + 'tau_2021_occu.csv').sort_values(by=['variable'])
dfepsN   = pd.read_csv(data_dir + 'epsN_2021.csv', index_col=0).sort_index(axis=1)
dfcurlyL = pd.read_csv(data_dir + 'curlyL_2021.csv', index_col=0).sort_index()
dfA      = pd.read_csv(data_dir + f'A{A_spec}.csv')
sectors  = dfA['short_names']
dfDemand = pd.read_csv(data_dir + 'demand_tab.csv')
dfLshare = pd.read_csv(data_dir + 'labor_tab.csv')
dfLabor_market_yearly= pd.read_csv(data_dir + 'uvh_annual_updated_occu.csv')
dfLabor_market_yearly = dfLabor_market_yearly.sort_values(by=['Year', 'variable'])
dfLabor_market_yearly = dfLabor_market_yearly.dropna(axis=0)
dfLabor_market_yearly = dfLabor_market_yearly[dfLabor_market_yearly['Year'] == 2021]
dfMatching_params = pd.read_csv(data_dir + 'matching_param_estimates_occu.csv')
dfL = pd.read_csv(data_dir + 'L_2021.csv')
shares = pd.read_csv(data_dir + 'energy_capital_shares.csv')
sector_names = list(dfA['short_names']) + ['Agg Y']
occupation_names = list(dfcurlyL.index)

# reformatting parameters
Omega = np.array(dfA.iloc[:, 1:], dtype='float64')
rescaler = np.matrix(1 - shares['Capital share'] - shares['Energy share'])
J = Omega.shape[0]
Omega = np.multiply(Omega, (np.repeat(rescaler, J).reshape(J, J)))
Psi = np.linalg.inv(np.eye(Omega.shape[0])-Omega)
curlyL = np.array(dfcurlyL)

O = dfcurlyL.shape[0]

epsN = np.array((np.array(dfLshare[f'labor_elasticity{A_spec}'], dtype='float64') * dfepsN.T).T)
epsN = np.multiply(epsN, rescaler.T)
# normalized epsN to back out sectoral tightness
epsN_norm = np.array(dfepsN)
epsD = np.array(dfDemand['demand_elasticity']).reshape((J,1))
epsK = np.matrix(shares[['Capital share', 'Energy share']])
K = epsK.shape[1]
dfLUmerged = dfLabor_market_yearly.merge(dfL, how='inner', left_on='variable',right_on='OCC_TITLE')

# if you want to turn off network linkages, uncomment these two lines of code.
# Omega = np.zeros_like(Omega)
# Psi = np.eye(Omega.shape[0])
θ = dfLabor_market_yearly['Tightness']
ν = dfMatching_params['unemployment_elasticity']
U = 1000*np.array(dfLabor_market_yearly['Unemployment']).reshape((O,1))
u = np.diag(dfLabor_market_yearly['u'])
#V = np.array(dfLabor_market_yearly['Vacancy']).reshape((O,1))
V = 1000*np.array(dfLabor_market_yearly['Vacancy']).reshape((O,1))
L_mat = np.array(dfLUmerged.iloc[:, 10:]).reshape((O,J))
L = np.sum(L_mat,1).reshape((O,1))

tau = dfTau['Tau']
curlyT = np.diag(tau)
curlyQ = np.diag(-ν)
curlyF = np.eye(O) + curlyQ
curlyF = u @ curlyF
theta = np.diag(V.flatten()/U.flatten())

#phi = np.diag(dfMatching_params['matching_efficiency'])
phi = np.exp(np.log(1000*np.array(dfLabor_market_yearly['Hires'])) + np.diag(curlyQ) * np.log(U.flatten()) - (1+np.diag(curlyQ)) * np.log(V.flatten()))

#Cobb-Douglas assumptions
dlog_lam = np.zeros((J,1))
dlog_epsN = np.zeros_like(epsN)
dlog_epsD = np.zeros_like(epsD)
curlyE = multi_occupation_network.curlyEFunc(dlog_epsN,epsN)

#### Picking Wage Assumption ####
WageAssumption = ['Labor Market Frictions + Production Linkages','Production Linkages Only', 'Labor Market Frictions Only']
####Setting up shock ####
sec_to_shock = 'dur'
shock_size = 0.01
shock_ind = sectors[sectors=='dur'].index.values[0]
# For reference, these are other sectors we can shock
print(sectors)
sec_dict = pd.read_excel("../data/raw/long_short_names_crosswalk.xlsx")
sec_full = sec_dict['Industry'][sec_dict['short_names'] == sec_to_shock].iloc[0].title()
print(f'the full name for {sec_to_shock} is {sec_full}')

sectorY_vec = np.zeros((J+1, len(WageAssumption)))
occT_vec = np.zeros((O+1, len(WageAssumption)))
occU_vec = np.zeros((O+1, len(WageAssumption)))
occUrate_vec = np.zeros((O+1, len(WageAssumption)))
dlog_A = np.zeros((J,1))
dlog_H = np.zeros((O,1))
dlog_K = np.zeros((K,1))

dlog_A[shock_ind] = shock_size

########## 4.4 Figures #######################
#### Generating Model Predicted Responses ####

#1
i = WageAssumption.index('Production Linkages Only')
gamma_A = 1
gamma_H = 1
gamma_K = 1
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFunc(gamma_A, gamma_H, gamma_K, Psi, curlyL, epsN, epsK)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))


#2
i = WageAssumption.index('Labor Market Frictions Only')
epsK_no_network = epsK.copy()
epsK_no_network[:,0] = np.sum(Omega,axis=1).reshape((J,1))+epsK_no_network[:,1]
gamma = 0.7
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, np.eye(J), epsN, epsK_no_network, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, np.eye(J), Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK_no_network)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, np.eye(J), Omega, curlyQ, curlyF, epsN, epsK_no_network, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

#3
i = WageAssumption.index('Labor Market Frictions + Production Linkages')
gamma = 0.7
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))



#To caluclate changes in the unemployement rate
#(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,0].flatten()) * U.flatten()/(U.flatten()+L.flatten()))*100

#### Creating Figures ####
#fig1
reorder = True
fig_seq = False
contains_agg = True
sector_names = list(dfA['short_names']) + ['Agg Y']
title = f'Response to 1% Technology Shock in {sec_full}'
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'../output/figures/paper/sec4/{sec_to_shock}_AshockY'
labels = WageAssumption
bar_plot(100*sectorY_vec, sector_names, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange','tab:green'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg, order_ascending=False)

if do_unemployment:
    #fig2
    occupation_names1 =  occupation_names + ['Agg $\\theta$']
    xlab = ''
    ylab = '$d \log \\theta$  (pct.)'
    save_path = f'../output/figures/paper/sec4/{sec_to_shock}_AshockT'
    labels = ['Labor Market Frictions + Production Linkages', 'Labor Market Frictions Only', 'Production Linkages Only']
    bar_plot(100*occT_vec[:,[0, 2, 1]], occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:green','tab:orange'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg, order_ascending=False)
    i = WageAssumption.index('Labor Market Frictions Only')
    print(pd.DataFrame({'name': occupation_names1, 'T':occT_vec[:, i]}))
    i = WageAssumption.index('Labor Market Frictions + Production Linkages')
    print(pd.DataFrame({'name': occupation_names1, 'T':occT_vec[:, i]}))

    # fig3
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = '$d \log U$  (pct.)'
    save_path = f'../output/figures/paper/sec4/{sec_to_shock}_AshockU'
    labels = ['Labor Market Frictions + Production Linkages', 'Labor Market Frictions Only', 'Production Linkages Only']
    bar_plot(100*occU_vec[:,[0,2,1]], occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:green','tab:orange'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg)

    # fig4
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = 'Pct. Point Change in Unemployment Rate'
    save_path = f'../output/figures/paper/sec4/{sec_to_shock}_AshockUrate'
    labels = ['Labor Market Frictions + Production Linkages', 'Labor Market Frictions Only', 'Production Linkages Only']
    bar_plot(100*occUrate_vec[:,[0,2,1]], occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:green','tab:orange'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg)


##################### 4.5 Figures ##############################
WageAssumption = ['Rigid Real', '0.7MP', '0.9MP', 'Rigid in Production Only', 'Rigid Nominal']
sec_to_shock = 'dur'
shock_size = 0.01
shock_ind = sectors[sectors=='dur'].index.values[0]
# For reference, these are other sectors we can shock
print(sectors)
sec_dict = pd.read_excel("../data/raw/long_short_names_crosswalk.xlsx")
sec_full = sec_dict['Industry'][sec_dict['short_names'] == sec_to_shock].iloc[0].title()
print(f'the full name for {sec_to_shock} is {sec_full}')

sectorY_vec = np.zeros((J+1, len(WageAssumption)))
occT_vec = np.zeros((O+1, len(WageAssumption)))
occU_vec = np.zeros((O+1, len(WageAssumption)))
occUrate_vec = np.zeros((O+1, len(WageAssumption)))
dlog_A = np.zeros((J,1))
dlog_H = np.zeros((O,1))
dlog_K = np.zeros((K,1))

dlog_A[shock_ind] = shock_size

i = WageAssumption.index('Rigid Real')
gamma = 0
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

i = WageAssumption.index('Rigid Nominal')
num=0
dlog_p = nominal_rigidity_network.PriceFunc(dlog_A=dlog_A, dlog_H=dlog_H, dlog_K=dlog_K, Psi=Psi, epsN=epsN, epsK=epsK, curlyQ=curlyQ, curlyT=curlyT, curlyF=curlyF, curlyL=curlyL, num=num)
dlog_theta = nominal_rigidity_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_p, Psi, epsN, epsK, curlyL, curlyQ, curlyT, curlyF, num=num)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

i = WageAssumption.index('0.7MP')
gamma = 0.7
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

i = WageAssumption.index('0.9MP')
gamma = 0.9
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

i = WageAssumption.index('Rigid in Production Only')
gamma_A = 1
gamma_H = 1
gamma_K = 1
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFunc(gamma_A, gamma_H, gamma_K, Psi, curlyL, epsN, epsK)
gamma = 0
epsW_A2, epsW_H2, epsW_K2 = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
epsW_A[15,:], epsW_H[15,:], epsW_K[15,:] = epsW_A2[15,:], epsW_H2[15,:], epsW_K2[15,:]

dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))


### Figures ###
#### Creating Figures ####
#fig1
sector_names = list(dfA['short_names']) + ['Agg Y']
title = f'Response to 1% Technology Shock in {sec_full}'
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'../output/figures/paper/sec4robust/{sec_to_shock}_AshockY'
labels = WageAssumption
bar_plot(100*sectorY_vec, sector_names, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'], rotation=30, fontsize=10, barWidth = 0.19, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg, order_ascending=False)

if do_unemployment:
#fig2
    occupation_names1 =  occupation_names + ['Agg $\\theta$']
    xlab = ''
    ylab = '$d \log \\theta$  (pct.)'
    save_path = f'../output/figures/paper/sec4robust/{sec_to_shock}_AshockT'
    labels = WageAssumption
    bar_plot(100*occT_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'], rotation=30, fontsize=10, barWidth = 0.19, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg, order_ascending=False)

    # fig3
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = '$d \log U$  (pct.)'
    save_path = f'../output/figures/paper/sec4robust/{sec_to_shock}_AshockU'
    labels = WageAssumption
    bar_plot(100*occU_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'], rotation=30, fontsize=10, barWidth = 0.19, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg)

    # fig4
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = 'Pct. Point Change in Unemployment Rate'
    save_path = f'../output/figures/paper/sec4robust/{sec_to_shock}_AshockUrate'
    labels = WageAssumption
    bar_plot(100*occUrate_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'], rotation=30, fontsize=10, barWidth = 0.19, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, contains_agg=contains_agg)


#################### 5 Figures #################################

dlog_A = np.zeros((J, 1))
dlog_H = np.zeros((O, 1))
dlog_K = np.zeros((K, 1))
dlog_K[1] = -0.01

service_sectors = ['Care', 'Clean', 'Food S', 'Trans', 'Soc S', 'Sales', 'Prot S']
shock_size = -0.02

for i in service_sectors:
    dlog_H[occupation_names.index(i)] = shock_size

dlog_lam = np.zeros_like(dlog_A)
dlog_epsN = np.zeros_like(epsN)
dlog_epsD = np.zeros_like(epsD)
num = 3
curlyE = multi_occupation_network.curlyEFunc(dlog_epsN,epsN)
WageAssumption = ['Labor Market Frictions + Production Linkages','Production Linkages Only']
sectorY_vec = np.zeros((J+1,len(WageAssumption)))
secT_vec = np.zeros((J+1,len(WageAssumption)))
occT_vec = np.zeros((O+1, len(WageAssumption)))
occU_vec = np.zeros((O+1, len(WageAssumption)))
occUrate_vec = np.zeros((O+1, len(WageAssumption)))
p_vec = np.zeros((J, len(WageAssumption)))


#1
i = WageAssumption.index('Production Linkages Only')
gamma_A = 1
gamma_H = 1
gamma_K = 1
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFunc(gamma_A, gamma_H, gamma_K, Psi, curlyL, epsN, epsK)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))
dlog_r = multi_occupation_network.rFunc(dlog_y, dlog_K, num)
dlog_p = multi_occupation_network.PriceFunc(dlog_A, dlog_r, dlog_wR, dlog_theta, Psi, curlyQ, epsN, epsK, curlyT, curlyL, num=num)
p_vec[:,i] = dlog_p.flatten()

#3
i = WageAssumption.index('Labor Market Frictions + Production Linkages')
gamma = 0
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
sectorY_vec[:-1, i] = dlog_y.flatten()
sectorY_vec[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
if do_unemployment:
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
    occU_vec[:-1,i] = dlog_U.flatten()
    occU_vec[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    occT_vec[-1, i] = multi_occupation_network.AggThetaFunc(dlog_theta, dlog_U, U, V)
    occUrate_vec[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    occUrate_vec[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))
dlog_r = multi_occupation_network.rFunc(dlog_y, dlog_K, num)
dlog_p = multi_occupation_network.PriceFunc(dlog_A, dlog_r, dlog_wR, dlog_theta, Psi, curlyQ, epsN, epsK, curlyT, curlyL, num=num)
p_vec[:,i] = dlog_p.flatten()


#To caluclate changes in the unemployement rate
#(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec[:-1,0].flatten()) * U.flatten()/(U.flatten()+L.flatten()))*100

#### Creating Figures ####
#fig1
sector_names = list(dfA['short_names']) + ['Agg Y']
title = f'Response to 1% Technology Shock in {sec_full}'
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'../output/figures/paper/sec5/energy_servshockY'
labels = WageAssumption
bar_plot(100*sectorY_vec, sector_names, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.4, dpi=300, reorder=False, order_ascending=False)

#fig2
occupation_names1 =  occupation_names + ['Agg $\\theta$']
xlab = ''
ylab = '$d \log \\theta$  (pct.)'
save_path = f'../output/figures/paper/sec5/energy_servshockT'
labels = ['Labor Market Frictions + Production Linkages', 'Production Linkages Only']
bar_plot(100*occT_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.4, dpi=300, reorder=False, order_ascending=False)

# fig3
occupation_names1 = occupation_names + ['Agg U']
xlab = ''
ylab = '$d \log U$  (pct.)'
save_path = f'../output/figures/paper/sec5/energy_servshockU'
labels = ['Labor Market Frictions + Production Linkages', 'Production Linkages Only']
bar_plot(100*occU_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.4, dpi=300, reorder=False, order_ascending=False)

# fig4
occupation_names1 = occupation_names + ['Agg U']
xlab = ''
ylab = 'Pct. Point Change in Unemployment Rate'
save_path = f'../output/figures/paper/sec5/energy_servshockUrate'
labels = ['Labor Market Frictions + Production Linkages', 'Production Linkages Only']
bar_plot(100*occUrate_vec, occupation_names1, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.4, dpi=300, reorder=False, order_ascending=False)

# fig5
xlab = ''
ylab = '$d\log p$  (pct.)'
save_path = f'../output/figures/paper/sec5/energy_servshockP'
labels = ['Labor Market Frictions + Production Linkages', 'Production Linkages Only']
bar_plot(100*p_vec, sectors, title, xlab, ylab, labels, save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.4, dpi=300,  contains_agg=False, reorder=False, order_ascending=False)


print('done')



