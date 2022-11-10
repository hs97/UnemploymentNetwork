import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from functions.helpers_labor_market import gen_curlyF_CD, gen_elasticity_Qtheta_CD, gen_tau_CD
from functions.cobb_douglas_network import cobb_douglas_network
import pandas as pd

##### loading data #####
data_dir = 'data/clean/'
dfA      = pd.read_csv(data_dir + 'A.csv')
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


##### reformatting parameters #####
Omega = np.array(dfA.iloc[:, 1:], dtype='float64')
phi = np.array(dfParam.φ).reshape(Omega.shape[0],1)
λ = np.array(dfParam.λ).reshape(Omega.shape[0],1)
elasticity_fN = np.array(dfParam.α).reshape(Omega.shape[0],1) #Chosen to ensure constant returns to scale. What if we pick them instead to allow for at most constant returns to scale and to equalize marginal product of labor at T=0 given actual labor at that time? What if instead we recalculate them each period to keep marginal product of labor constant across industries?
elasticity_Dc = np.array(dfParam.θ_alt).reshape(Omega.shape[0],1)
elasticity_Dc = elasticity_Dc/np.sum(elasticity_Dc)
η = 0.5
eta = np.ones_like(elasticity_fN) * η 
s = 0.03 * np.ones_like(elasticity_fN) # exogenous separation rate (currently set to arbitrary value)
r = 0.8 * np.ones_like(elasticity_fN) # recruiting cost in relative wage units (currently set to arbitrary value)

#### Deriving required labor market measures #####
Nsectors = Omega.shape[0]
H = np.array(dfLabor_market_yearly.u[-Nsectors-1:-1]+dfLabor_market_yearly.e[-Nsectors-1:-1]).reshape(Nsectors,1)
L = np.array(dfLabor_market_yearly.e[-Nsectors-1:-1]).reshape(Nsectors,1)
theta = np.array(dfLabor_market_yearly.v[-Nsectors-1:-1]/dfLabor_market_yearly.u[-Nsectors-1:-1]).reshape(Nsectors,1)
U = H - L

elasticity_Qtheta = gen_elasticity_Qtheta_CD(theta,eta)
curlyF = gen_curlyF_CD(theta,eta,phi,s)
tau = gen_tau_CD(theta,eta,phi,s,r)

#### Setting up production network ####
## Rigid nominal wages ##
elasticity_wtheta = np.zeros_like(Omega)
elasticity_wA = np.zeros_like(Omega)
cobb_douglas_rigid_nominal = cobb_douglas_network(Omega, elasticity_Dc, elasticity_fN, elasticity_Qtheta, tau, curlyF, elasticity_wtheta, elasticity_wA)
# Shocks
dlogA = np.zeros_like(elasticity_Dc)
dlogH = np.zeros_like(elasticity_Dc)

# Shock to technology
dlogA = -0.01 * np.ones_like(dlogA) # shock to all sectors
#dlogA[-2] = -0.05 # shock to transportation
tech_shock_nominal = cobb_douglas_rigid_nominal.shocks(dlogA,dlogH,H,L,U) #doesn't seem to be working because negative shock to productivity has large negative effect on output in all sectors.

## Wages move with aggregate price level ##
gamma = 0.25
I = np.eye(Omega.shape[0])
term1 = I - (gamma*np.ones_like(elasticity_Dc)) @ elasticity_Dc.T @ cobb_douglas_rigid_nominal.Psi @ np.diag(elasticity_fN.flatten())
Inv_term = np.linalg.inv(term1)
elasticity_wA = -Inv_term @ (gamma * np.ones_like(elasticity_Dc)) @ elasticity_Dc.T @ cobb_douglas_rigid_nominal.Psi 
elasticity_wtheta = elasticity_wA @ np.diag(elasticity_fN.flatten()) @ np.diag(tau.flatten()) @ np.diag(elasticity_Qtheta.flatten())
cobb_douglas_rigid_real = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
tech_shock_real = cobb_douglas_rigid_real.shocks(dlogA,dlogH,H,L,U) 

## Wages move with sectoral price level ##
I = np.eye(Omega.shape[0])
term1_sectoral = I - (gamma*cobb_douglas_rigid_nominal.Psi) @ np.diag(elasticity_fN.flatten())
Inv_term = np.linalg.inv(term1_sectoral)
elasticity_wA = -Inv_term @ (gamma*cobb_douglas_rigid_nominal.Psi)
elasticity_wtheta = elasticity_wA @ np.diag(elasticity_fN.flatten()) @ np.diag(tau.flatten()) @ np.diag(elasticity_Qtheta.flatten())
cobb_douglas_sectoral_real = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
tech_shock_sectoral = cobb_douglas_sectoral_real.shocks(dlogA,dlogH,H,L,U) 

## Wages rise one for one with own sector A, do no change with tightness ##
elasticity_wA = np.eye(Omega.shape[0])
elasticity_wtheta = np.zeros_like(Omega)
cobb_douglas_eyeA = cobb_douglas_rigid_nominal.wage_elasticities(elasticity_wtheta, elasticity_wA)
tech_shock_eyeA = cobb_douglas_eyeA.shocks(dlogA,dlogH,H,L,U) 

###### Shock to size of labor force #######
dlogA = np.zeros_like(elasticity_Dc)
dlogH = -0.01*np.ones_like(elasticity_Dc)
H_shock_nominal = cobb_douglas_rigid_nominal.shocks(dlogA,dlogH,H,L,U)

# Shock to size of labor force
dlogA = np.zeros_like(elasticity_Dc)
dlogH = -0.01*np.ones_like(elasticity_Dc)
H_shock_real = cobb_douglas_rigid_real.shocks(dlogA,dlogH,H,L,U)

# Shock to size of labor force
dlogA = np.zeros_like(elasticity_Dc)
dlogH = -0.01*np.ones_like(elasticity_Dc)
H_shock_sectoral = cobb_douglas_sectoral_real.shocks(dlogA,dlogH,H,L,U)

# Shock to size of labor force
dlogA = np.zeros_like(elasticity_Dc)
dlogH = -0.01*np.ones_like(elasticity_Dc)
H_shock_eyeA = cobb_douglas_eyeA.shocks(dlogA,dlogH,H,L,U)

#### Figures ####

print('done')