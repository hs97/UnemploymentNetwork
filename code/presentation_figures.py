import numpy as np
import pandas as pd
import functions.multi_occupation_network as multi_occupation_network
import functions.nominal_rigidity_network as nominal_rigidity_network
import matplotlib.pyplot as plt
from  functions.network_figures import bar_plot
import functions.no_network_reparametrization as nonet
import scipy.optimize as opt

#### Reading in calibration params ####
do_unemployment = 1
A_spec = 1
data_dir = 'data/clean/'
dfTau = pd.read_csv(data_dir + 'tau_2021_occu.csv').sort_values(by=['variable'])
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
V = np.array(dfLabor_market_yearly['Vacancy']).reshape((O,1))
L_mat = np.array(dfLUmerged.iloc[:, 7:]).reshape((O,J))
L = np.sum(L_mat,1).reshape((O,1))

tau = dfTau['Tau']
curlyT = np.diag(tau)
curlyQ = np.diag(-ν)
curlyF =  np.eye(O) + curlyQ
theta = np.diag(V.flatten()/U.flatten())
print(theta)

phi = np.diag(dfMatching_params['matching_efficiency'])

#Cobb-Douglas assumptions
dlog_lam = np.zeros((J,1))
dlog_epsN = np.zeros_like(epsN)
dlog_epsD = np.zeros_like(epsD)
curlyE = multi_occupation_network.curlyEFunc(dlog_epsN,epsN)

# Setting which sectors to shock, and getting the full name of the sector. 
sec_to_shock = 'dur'
shock_size = 0.01
shock_ind = sectors[sectors=='dur'].index.values[0]
# For reference, these are other sectors we can shock
print(sectors)
sec_dict = pd.read_excel("data/raw/long_short_names_crosswalk.xlsx")
sec_full = sec_dict['Industry'][sec_dict['short_names'] == sec_to_shock].iloc[0].title()
print(f'the full name for {sec_to_shock} is {sec_full}')


#### Picking Wage Assumption ####
WageAssumption = ['Labor Market Frictions + Production Linkages', 'Production Linkages Only', 'Labor Market Frictions Only']
####Setting up shock ####
sec_to_shock = 'dur'
shock_size = 0.01
shock_ind = sectors[sectors=='dur'].index.values[0]
# For reference, these are other sectors we can shock
print(sectors)
sec_dict = pd.read_excel("data/raw/long_short_names_crosswalk.xlsx")
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
#gamma_A = 0.7
#gamma_H = 0
#gamma_K = 0
#epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFunc(gamma_A, gamma_H, gamma_K, np.eye(J), curlyL, epsN, epsK_no_network)
dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, np.eye(J), np.zeros_like(Psi), curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK_no_network)
occT_vec[:-1, i] = dlog_theta.flatten()
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, np.eye(J), np.zeros_like(Psi), curlyQ, curlyF, epsN, epsK_no_network, curlyT, curlyE)
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
epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, np.eye(J), epsN, epsK_no_network, curlyF, curlyQ, curlyT, curlyL)
#gamma_A = 0.7
#gamma_H = 0
#gamma_K = 0
#epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFunc(gamma_A, gamma_H, gamma_K, Psi, curlyL, epsN, epsK)
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

reorder = True
fig_seq = True
contains_agg = True
#fig1
sector_names = list(dfA['short_names']) + ['Agg Y']
title = f'Response to 1% Technology Shock in {sec_full}'
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'output/figures/presentation/{sec_to_shock}_AshockY'
labels = WageAssumption
bar_plot(100*sectorY_vec, sector_names, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange','tab:green'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=False, contains_agg=contains_agg)

# Fix by removing the no frictions case from here
if do_unemployment:
    #fig2
    occupation_names1 =  occupation_names + ['Agg $\\theta$']
    xlab = ''
    ylab = '$d \log \\theta$  (pct.)'
    save_path = f'output/figures/presentation/{sec_to_shock}_AshockT'
    labels = WageAssumption
    bar_plot(100*occT_vec, occupation_names1, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange','tab:green'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=False, contains_agg=contains_agg)

    # fig3
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = '$d \log U$  (pct.)'
    save_path = f'output/figures/presentation/{sec_to_shock}_AshockU'
    labels = WageAssumption
    bar_plot(100*occU_vec, occupation_names1, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange','tab:green'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=True, contains_agg=contains_agg)

    # fig4
    occupation_names1 = occupation_names + ['Agg U']
    xlab = ''
    ylab = 'Pct. Point Change in Unemployment Rate'
    save_path = f'output/figures/presentation/{sec_to_shock}_AshockUrate'
    labels = WageAssumption
    bar_plot(100*occUrate_vec, occupation_names1, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange','tab:green'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=True, contains_agg=contains_agg)


## Additional figures

# Matching propagation with matching params only
reparamY_vec = np.zeros((J+1,2))
reparamY_vec[:,0] = sectorY_vec[:,0]

param = {'epsN':epsN, 'curlyL':curlyL, 'dlog_A':dlog_A, 'dlog_H':dlog_H, 'dlog_K':dlog_K, 'dlog_wR':dlog_wR, 'dlog_epsN':dlog_epsN,
         'dlog_lam':dlog_lam, 'curlyF':curlyF, 'curlyQ':curlyQ, 'curlyE':curlyE, 'curlyT':curlyT, 'dlog_epsD':dlog_epsD,
         'epsD':epsD,'epsK':epsK, 'U':U, 'L':L, 'Psi':Psi, 'Omega':Omega ,'agg':True, 'targ':'y', 'close_params':False, 'control_name':'curlyQ'}
if param['control_name'] == 'curlyT':
    control0 = np.diag(curlyT)
if param['control_name'] == 'curlyQ':
    control0 = np.diag(curlyQ)
optim_out = opt.minimize(fun=nonet.obj_reparam_tau, x0=control0, args=(param), method='Nelder-Mead', tol=1e-8, options={'maxiter':10000})
if param['control_name'] == 'curlyT':
    curlyTtil = np.diag(optim_out.x)
    curlyQtil = curlyQ
    curlyFtil = curlyF
if param['control_name'] == 'curlyQ':
    curlyTtil = curlyT
    curlyQtil = np.diag(optim_out.x)
    curlyFtil = np.eye(O) + curlyQtil

I_j = np.eye(curlyL.shape[1])
Psi_til = I_j
Omega_til = np.zeros_like(Psi)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi_til, Omega_til, curlyFtil, curlyQtil, curlyTtil, curlyE, curlyL, epsN, epsK)
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi_til, Omega_til, curlyQtil, curlyFtil, epsN, epsK, curlyTtil, curlyE)
reparamY_vec[:-1, 1] = dlog_y.flatten()
reparamY_vec[-1, 1] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)

# fig reparam
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'output/figures/presentation/{sec_to_shock}_AshockYreparam'
labels = ['Labor Market Frictions + Production Linkages', 'Reparametrized Labor Market Frictions']
bar_plot(100*reparamY_vec, sector_names, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.45, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=False, contains_agg=contains_agg)

if do_unemployment:
    reparamU_vec = np.zeros((O+1,2))
    reparamU_vec[:,0] = occU_vec[:,0]
    reparamUrate_vec = np.zeros((O+1,2))
    reparamUrate_vec[:,0] = occUrate_vec[:,0]
    param['targ'] = 'U'
    param['control_name'] = 'curlyQ'
    if param['control_name'] == 'curlyT':
        control0 = np.diag(curlyT)
    if param['control_name'] == 'curlyQ':
        control0 = np.diag(curlyQ)
    optim_out = opt.minimize(fun=nonet.obj_reparam_tau, x0=control0, args=(param), method='Nelder-Mead', tol=1e-8, options={'maxiter':10000})
    if param['control_name'] == 'curlyT':
        curlyTtil = np.diag(optim_out.x)
        curlyQtil = curlyQ
        curlyFtil = curlyF
    if param['control_name'] == 'curlyQ':
        curlyTtil = curlyT
        curlyQtil = np.diag(optim_out.x)
        curlyFtil = np.eye(O) + curlyQtil
    I_j = np.eye(curlyL.shape[1])
    Psi_til = I_j
    Omega_til = np.zeros_like(Psi)
    dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi_til, Omega_til, curlyFtil, curlyQtil, curlyTtil, curlyE, curlyL, epsN, epsK)
    dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi_til, Omega_til, curlyQtil, curlyFtil, epsN, epsK, curlyTtil, curlyE)
    reparamY_vec[:-1, 1] = dlog_y.flatten()
    reparamY_vec[-1, 1] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
    dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyFtil, U, L)
    reparamU_vec[:-1,1] = dlog_U.flatten()
    reparamU_vec[-1, 1] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
    reparamUrate_vec[:-1,1] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+reparamU_vec[:-1,1].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
    reparamUrate_vec[-1,1] = -(U.sum()/(U.sum()+L.sum()) - (1+reparamU_vec[-1,1].flatten()) * U.sum()/(U.sum()+L.sum()))
    # fig reparam
    xlab = ''
    ylab = 'Pct. Point Change in Unemployment Rate'
    save_path = f'output/figures/presentation/{sec_to_shock}_AshockUratereparam'
    occupation_names1 = occupation_names + ['Agg U']
    labels = ['Labor Market Frictions + Production Linkages', 'Reparametrized Labor Market Frictions']
    bar_plot(100*reparamUrate_vec, occupation_names1, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.45, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=True, contains_agg=contains_agg)
    
    # fig reparam 
    xlab = ''
    ylab = '$\ d\log y$ (pct.)'
    save_path = f'output/figures/presentation/{sec_to_shock}_AshockUratereparamY'
    labels = ['Labor Market Frictions + Production Linkages', 'Reparametrized Labor Market Frictions']
    bar_plot(100*reparamY_vec, sector_names, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:orange'], rotation=30, fontsize=10, barWidth = 0.45, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=False, contains_agg=contains_agg)
            

I_j = np.eye(curlyL.shape[1])
Psi_til = I_j
Omega_til = np.zeros_like(Psi)
dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi_til, Omega_til, curlyFtil, curlyQtil, curlyTtil, curlyE, curlyL, epsN, epsK)
dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi_til, Omega_til, curlyQtil, curlyFtil, epsN, epsK, curlyTtil, curlyE)
reparamY_vec[:-1, 1] = dlog_y.flatten()
reparamY_vec[-1, 1] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)

# What is unemployment pre- and post-shock
Urate_data = np.zeros_like(occUrate_vec[:,0])

Urate_data[:-1] = U.flatten()/(U.flatten()+L.flatten())
Urate_data[-1] = np.sum(U.flatten())/np.sum((U.flatten()+L.flatten()))
Urate_comp = np.zeros((O+1,3))
Urate_comp[:,0]  = occUrate_vec[:,0].reshape(O+1,)
Urate_comp[:,1]  = Urate_data.reshape(O+1,)
Urate_comp[:,2]  = Urate_data.reshape(O+1,) + occUrate_vec[:,0].reshape(O+1,)
Urate_compoccupation_names1 = occupation_names + ['Agg U']

xlab = ''
ylab = 'Pct. Point Change in Unemployment Rate'
save_path = f'output/figures/presentation/{sec_to_shock}_UrateComp'
labels = ['Change in Unemployment Rate', 'Pre-Shock Unemployment Rate', 'Post-Shock Unemployment Rate']
bar_plot(100*Urate_comp, occupation_names1, title, xlab, ylab, labels, save_path=save_path, colors=['tab:blue','tab:green','tab:orange'], rotation=30, fontsize=10, barWidth = 0.3, dpi=300, reorder=reorder, gen_fig_sequence=fig_seq, order_ascending=True, contains_agg=contains_agg)


# Robustness to wage rigidity
MP_names = ['0.7MP', '0.9MP', '0.95MP', '0.99MP']
sectorY_vec_robust = np.zeros((sectorY_vec.shape[0],4))
occU_vec_robust = np.zeros((occU_vec.shape[0],4))
occUrate_vec_robust = np.zeros_like(occU_vec_robust)

MP_vec = [0.7, 0.9, 0.95, 0.99]
for i in range(len(MP_vec)):
    gamma = MP_vec[i]
    epsW_A, epsW_H, epsW_K = multi_occupation_network.WageElasticityFuncMP(gamma, np.eye(J), epsN, epsK_no_network, curlyF, curlyQ, curlyT, curlyL)

    dlog_wR = multi_occupation_network.WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K)
    dlog_theta = multi_occupation_network.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
    dlog_y = multi_occupation_network.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
    sectorY_vec_robust[:-1, i] = dlog_y.flatten()
    sectorY_vec_robust[-1, i] = multi_occupation_network.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
    if do_unemployment:
        dlog_U = multi_occupation_network.UnemploymentFunc(dlog_theta, dlog_H, curlyF, U, L)
        occU_vec_robust[:-1,i] = dlog_U.flatten()
        occU_vec_robust[-1, i] = multi_occupation_network.AggUnemploymentFunc(dlog_U, U)
        occUrate_vec_robust[:-1,i] = -(U.flatten()/(U.flatten()+L.flatten()) - (1+occU_vec_robust[:-1,i].flatten()) * U.flatten()/(U.flatten()+L.flatten()))
        occUrate_vec_robust[-1,i] = -(U.sum()/(U.sum()+L.sum()) - (1+occU_vec_robust[-1,i].flatten()) * U.sum()/(U.sum()+L.sum()))

# fig robust
xlab = ''
ylab = '$\ d\log y$ (pct.)'
save_path = f'output/figures/presentation/{sec_to_shock}_AshockY_robust'
bar_plot(100*sectorY_vec_robust, sector_names, title, xlab, ylab, MP_names, save_path=save_path, colors=['tab:blue','tab:orange','tab:green','tab:red'], rotation=30, fontsize=10, barWidth = 0.23, dpi=300, reorder=reorder, gen_fig_sequence=False, order_ascending=False, contains_agg=contains_agg)

occupation_names1 = occupation_names + ['Agg U']
xlab = ''
ylab = 'Pct. Point Change in Unemployment Rate'
save_path = f'output/figures/presentation/{sec_to_shock}_AshockUrate_robust'
labels = WageAssumption
bar_plot(100*occUrate_vec_robust, occupation_names1, title, xlab, ylab, MP_names, save_path=save_path, colors=['tab:blue','tab:orange','tab:green','tab:red'], rotation=30, fontsize=10, barWidth = 0.23, dpi=300, reorder=reorder, gen_fig_sequence=False, order_ascending=True, contains_agg=contains_agg)

print('done')