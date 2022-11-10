import numpy as np

##### Tightness #####
def theta(dlogA, dlogH, curlyF, elasticity_Ldtheta, elasticity_LdA):
    curlyF = curlyF.reshape(curlyF.shape[0],)
    theta_coeff = np.linalg.inv(np.diag(curlyF)-elasticity_Ldtheta)
    dlogtheta = theta_coeff @ elasticity_LdA @ dlogA - theta_coeff @ dlogH
    return dlogtheta

##### elasticity of labor demand wrt theta ##### 
def elasticity_labor_demand(elasticity_fN, Psi, elasticity_wtheta, elasticity_lambdatheta, elasticity_elasticityfNtheta):
    I = np.eye(Psi.shape[0])
    elasticity_fN=elasticity_fN.reshape(elasticity_fN.shape[0],)
    elasticity_fN_coeff = np.linalg.inv(I-np.diag(elasticity_fN) @ Psi) @ np.diag(elasticity_fN) 
    elasticity_Ldtheta = elasticity_lambdatheta + (I + elasticity_fN_coeff) @ elasticity_elasticityfNtheta - elasticity_wtheta
    return elasticity_Ldtheta

##### Generating other necessary labor market inputs in case of Cobb-Douglas matching #####
# will need different functions here for different functional forma assumptions about matching

def gen_curlyF_CD(theta,eta,phi,s):
    return s/(s + phi*np.power(theta,1-eta)) * (1-eta)

def gen_elasticity_Qtheta_CD(theta, eta):
    return -eta * np.ones_like(theta) 

def gen_tau_CD(theta,eta,phi,s,r):
    return r * s / (phi * np.power(theta,-eta) - r * s)
