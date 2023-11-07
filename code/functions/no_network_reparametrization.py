import numpy as np
import scipy.optimize as opt
import pandas as pd
import functions.multi_occupation_network as multi

def obj_reparam_tau(control, param):
    epsN = param['epsN']
    curlyL = param['curlyL']
    dlog_A = param['dlog_A']
    dlog_H = param['dlog_H']
    dlog_K = param['dlog_K']
    dlog_wR = param['dlog_wR']
    dlog_epsN = param['dlog_epsN']
    dlog_lam = param['dlog_lam']
    curlyF = param['curlyF']
    curlyQ = param['curlyQ']
    curlyE = param['curlyE']
    curlyT = param['curlyT']
    dlog_epsD = param['dlog_epsD']
    epsD = param['epsD']
    epsK = param['epsK']
    U = param['U']
    L = param['L']
    Psi = param['Psi']
    Omega = param['Omega']
    targ = param['targ']
    agg = param['agg']
    close_params = param['close_params']
    control_name = param['control_name']
        
    I_o = np.eye(curlyL.shape[0])
    I_j = np.eye(curlyL.shape[1])
    Psi_til = I_j
    Omega_til = np.zeros_like(Psi)
    
    if control_name == 'curlyT':
        curlyTtil = np.diag(control)
        curlyQtil = curlyQ 
        curlyFtil = curlyF
    if control_name == 'curlyQ':
        curlyQtil = np.diag(control)
        curlyFtil  = np.eye(curlyQtil.shape[0]) + curlyQtil
        curlyTtil = curlyT
    
    dlog_theta = multi.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK)
    dlog_theta_til = multi.ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi_til, Omega_til, curlyFtil, curlyQtil, curlyTtil, curlyE, curlyL, epsN, epsK)
    if targ == 'y':
        dlog_y = multi.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE)
        dlog_y_til = multi.OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta_til, dlog_lam, Psi_til, Omega_til, curlyQtil, curlyFtil, epsN, epsK, curlyTtil, curlyE)
        if agg:
            Yagg = multi.AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD)
            Yagg_til = multi.AggOutputFunc(dlog_y_til, dlog_lam, dlog_epsD, epsD)
            obj = np.linalg.norm(Yagg-Yagg_til)
        else: 
            obj = np.linalg.norm(dlog_y-dlog_y_til)
    if targ == 'U':
        dlog_U = multi.UnemploymentFunc(dlog_theta, dlog_H, curlyFtil, U, L)
        dlog_U_til = multi.UnemploymentFunc(dlog_theta_til, dlog_H, curlyFtil, U, L)
        if agg:
            U_agg = multi.AggUnemploymentFunc(dlog_U, U)
            U_agg_til = multi.AggUnemploymentFunc(dlog_U_til, U)
            obj = np.linalg.norm(U_agg-U_agg_til)
        else:
            obj = np.linalg.norm(dlog_U - dlog_U_til)
    
    if close_params:
        return obj + np.linalg.norm(curlyT-curlyTtil) + np.linalg.norm(curlyQ-curlyQtil)
    else:
        return obj

    