import numpy as np
import scipy.optimize as opt
import pandas as pd

def obj_reparam(control, param):
    epsNtil = param['epsNtil']
    coeffA = param['coeffA']
    curlyL = param['curlyL']
    epsW_A = param['epsW_A']
    dlog_A = param['dlog_A']
    sigma = param['sigma']
    curlyMtil = param['curlyMtil']
    agg = param['agg']
    I_o = np.eye(curlyL.shape[0])
    I_j = np.eye(curlyL.shape[1])
    
    curlyTtil = np.diag(control)

    Xi_theta_till = curlyL @ I_j @ epsNtil @ (I_o - curlyMtil @ (I_o + curlyTtil))
    coeffAtil = I_j @ (I_j + epsNtil @ (I_o - curlyMtil @ (I_o + curlyTtil)) @ np.linalg.inv((I_o - curlyMtil - Xi_theta_till)) @ (curlyL @ I_j - epsW_A))
    if agg:
        return np.linalg.norm( sigma.T @ (coeffAtil - coeffA) @ dlog_A)
    else:
        return np.max(np.abs(100*(coeffAtil-coeffA) @ dlog_A))