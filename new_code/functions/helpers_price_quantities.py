import numpy as np

##### Prices ######
def pricing(dlogw, dlogtheta, dlogA, elasticity_fN, elasticity_Qtheta, tau, Psi):
    productivity_term = - Psi @ dlogA
    elasticity_fN=elasticity_fN.reshape(elasticity_fN.shape[0],)
    tau = tau.reshape(tau.shape[0],)
    elasticity_Qtheta = elasticity_Qtheta.reshape(elasticity_Qtheta.shape[0])
    labor_term = Psi @ np.diag(elasticity_fN) @ (dlogw - np.diag(tau) @ np.diag(elasticity_Qtheta) @ dlogtheta)
    dlogp = labor_term + productivity_term
    return dlogp

##### Wages #####
def wages(dlogtheta,dlogA,elasticity_wtheta, elasticity_wA):
    dlogw = elasticity_wtheta @ dlogtheta + elasticity_wA @ dlogA 
    return dlogw

##### Sectoral Output #####
def sectoral_output(dlogp,dloglambda,dlogelasticity_fN,elasticity_fN,Psi):
    I = np.eye(Psi.shape[0])
    elasticity_fN=elasticity_fN.reshape(elasticity_fN.shape[0],)
    elasticity_fN_coeff = np.linalg.inv(I-np.diag(elasticity_fN) @ Psi) @ np.diag(elasticity_fN) 
    dlogy = -dlogp + dloglambda + elasticity_fN_coeff @ dlogelasticity_fN
    return dlogy

##### Real Output #####
def aggregate_real_output(dlogp,dlogelasticity_fN,dlogelasticity_Dc,elasticity_Dc,elasticity_fN,Psi):
    I = np.eye(Psi.shape[0])
    elasticity_fN=elasticity_fN.reshape(elasticity_fN.shape[0],)
    elasticity_fN_coeff = np.linalg.inv(I-np.diag(elasticity_fN) @ Psi) @ np.diag(elasticity_fN) 
    dlogY = elasticity_Dc.T @ dlogp + elasticity_Dc.T @ (dlogelasticity_Dc + elasticity_fN_coeff @ dlogelasticity_fN)
    return dlogY

##### Aggregate Prices #####

