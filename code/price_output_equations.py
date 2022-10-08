import numpy as np
import pandas as pd

###### values for testing: picked arbitrarily ######
Omega   = np.zeros((3,3),dtype=np.float64)
Omega[0,:] = np.array([0,0.2,0.8]) #this is our theta from before
Omega[1,:] = np.array([0,0.3,0.5])
Omega[2,:] = np.array([0,0.15,0.4])

Omega_L = np.array([0,0.1,0.3],dtype=np.float64)
lam     = np.array([1,0.6,0.5])

elasticities = np.ones((3,2))
shocks = np.zeros_like(elasticities)
shocks[:,1] = np.array([0,0.5,-2])

###### Testing with our actual values ######
# loading data
data_dir = 'data/clean/'
dfA      = pd.read_csv(data_dir + 'A.csv')
dfParam  = pd.read_csv(data_dir + 'params.csv')

# reformatting parameters
A = np.array(dfA.iloc[:, 1:], dtype='float64')
λ = np.array(dfParam.λ)
θ = np.array(dfParam.θ)

# Creating Omega matrix and arbitrary elasticities/shock matrices
Omega = np.zeros((A.shape[0]+1,A.shape[1]+1))
Omega[1:,1:] = A 
Omega[0,1:]  = θ
lam = np.ones((λ.shape[0]+1))
lam[1:] = λ

elasticities = np.ones((Omega.shape[1],2))
shocks = np.zeros_like(elasticities)
shocks[5,1] = np.log(1.05) #5% positive shock to one sector

###### Helpers ######

def Theta_mat(Omega):
    return 1/np.sum(Omega,axis=1)

def Phi_mat(Omega,Theta):
    if Omega.shape == Theta.shape:
        return Theta * Omega
    else: 
        return np.tile(Theta,(Omega.shape[1],1)) * Omega

def Lambda_mat(lam, Omega):
    lam_intermediate = np.tile(lam,(Omega.shape[1],1)) / np.tile(lam,(Omega.shape[1],1)).T
    Lambda = lam_intermediate * Omega.T 
    gamma = np.sum(Lambda,axis=1)
    return Lambda, gamma

def Xi_mat(Phi,Theta,Lambda,gamma):
    I = np.eye(Phi.shape[0])
    P1 = np.linalg.inv(I - Phi) * np.tile(Theta - 1,(Phi.shape[1],1))
    P2 = np.linalg.inv(I - Lambda) @ Lambda - np.linalg.inv(I - Lambda) * np.tile(gamma,(Lambda.shape[1],1))
    return P1 @ P2, P2

###### Pricing Equation ######
def prices_quantities(Omega, lam, elasticities, shocks):
    I = np.eye(Omega.shape[1])
    Theta = Theta_mat(Omega)
    Phi   = Phi_mat(Omega,Theta)
    Lambda, gamma = Lambda_mat(lam,Omega)
    Xi, Xi_output = Xi_mat(Phi,Theta,Lambda,gamma)  
    dlog_p = -np.sum(np.linalg.inv((I-Phi) @ (I+Xi)) @ (elasticities * shocks),axis=1)
    dlog_x = Xi_output @ dlog_p
    return dlog_p, dlog_x


###### code testing ######
dlog_p, dlog_x = prices_quantities(Omega, lam, elasticities, shocks)

print('done')