import numpy as np
from numba import njit
import pandas as pd
from scipy.optimize import root

# cobb-douglas matching function and first derivative

def m_cd(u,v,φ,η):
    # takes np arrays ofvacancies, unemployment rates, matching efficiencies, and vacancy weight in matching function
    # returns number of matches
    U = np.power(u,1-η)
    V = np.power(v,η)
    return φ*V*U

def mu_cd(u,v,φ,η):
    # takes np arrays ofvacancies, unemployment rates, matching efficiencies, and vacancy weight in matching function
    # returns marginal increase matches per increase in unemployment
    τ = np.power(v/u,η)
    return φ*τ

def Lstar(u,v,e,φ,η,mfunc):
    # takes np arrays ofvacancies, unemployment rates, existing labor stocks, matching efficiencies, parameters of matching function, and a matching function
    # in cobb-douglas case, η is just the vacancy weight of vacancies, but can accomodate more general cases
    # realized labor in each sector
    return e + mfunc(u,v,φ,η)

def Lones(u,v,e,φ,η,mfunc):
    return np.ones(u.shape[0])

def objective(uopt,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc):
    
    #matching function componenets
    if np.any(uopt <= 0):
        obj = np.ones_like(v)*10000
    else:
        mu  = mufunc(uopt,v,φ,η)
        L   = Lfunc(uopt,v,e,φ,η,mfunc)
        FOC = λ*α*mu/L 
        
        obj      = np.empty_like(uopt)
        obj[:-1] = FOC[0] - FOC[1:] 
        obj[-1]  = np.sum(uopt+e) - 1

    return obj

def ustar(objective,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc,uguess=np.array([]),method='hybr',tol=1e-6,maxiter=1e4):
    #wrapper for scipy root in ustar notation
    if uguess.shape[0] == 0:
        uguess = np.zeros_like(v)+0.03
    us = root(objective,uguess,args=(v,e,φ,η,λ,α,mfunc,mufunc,Lfunc),method=method,tol=tol,options={'maxiter':int(maxiter)})
    return  us

# Mismatch index measures and optimal unemployment 
def Mindex(u,uopt,v,φ,η,mfunc):
    h    = mfunc(u,v,φ,η)
    hopt = mfunc(uopt,v,φ,η)
    return 1 - np.sum(h)/np.sum(hopt)

def ucounterfactual(u,uopt,Mfunc):
    return

