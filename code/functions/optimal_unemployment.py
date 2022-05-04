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

def ustar(objective,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc,uguess_mean=np.array([]),tol=1e-6,maxiter=1e4,ntrue=100,guessrange=0.1):
    #wrapper for scipy root in ustar notation, also implements robustness to intial guess with randomly generated intital guesses
    if uguess_mean.shape[0] == 0:
        uguess_mean = np.ones_like(v)*(1-np.sum(v+e))
    count_true = 0
    out_mat    = np.array([])
    itercount = 0
    while count_true<ntrue and itercount<maxiter:
        uguess = np.zeros_like(v)
        for i in range(v.shape[0]):
            uguess[i] = np.random.uniform(uguess_mean[i]-guessrange/2,uguess_mean[i]+guessrange/2,1)
        uguess = np.abs(uguess)
        us = root(objective,uguess,args=(v,e,φ,η,λ,α,mfunc,mufunc,Lfunc),method='hybr',tol=tol)
        count_true += us.success
        itercount  += 1
        if us.success == True:
            out_mat = np.append(out_mat,us.x)
            print('Num converged: ' + str(count_true))
    out_mat = out_mat.reshape((100,v.shape[0]))
    out_gap = out_mat - out_mat[0,:]
    if np.max(np.abs(out_gap))>1000*tol:
        success = False
    else:
        success = True
    return  np.mean(out_mat,axis=0), success

# Mismatch index measures and optimal unemployment 
def Mindex(u,uopt,v,φ,η,mfunc):
    h    = mfunc(u,v,φ,η)
    hopt = mfunc(uopt,v,φ,η)
    return 1 - np.sum(h)/np.sum(hopt)


# Function that runs code once for each time period in the data

def mismatch_estimation(df,objective,φ,η,λ,α,mfunc,mufunc,Lfunc,tol=1e-6,maxiter=1e5,ntrue=100,guessrange=0.1):
    output = pd.DataFrame(index=df.date.unique(),columns=df.BEA_sector.unique())
    M_t    = np.zeros(df.date.unique().shape[0]) 
       
    for i in range(df.date.unique().shape[0]):
        vraw = np.array(df.v[df.date==df.date.unique()[i]])
        uraw = np.array(df.u[df.date==df.date.unique()[i]])
        eraw = np.array(df.e[df.date==df.date.unique()[i]])
        e    = eraw/np.sum(eraw+uraw)
        u    = uraw/np.sum(eraw+uraw)
        v    = vraw/np.sum(eraw+uraw)
           
        ustar_t, success = ustar(objective,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc,uguess_mean=u,tol=tol,maxiter=maxiter,ntrue=ntrue,guessrange=guessrange)
        output.iloc[i,:] = ustar_t
        M_t[i]  = Mindex(u,ustar_t,v,φ,η,mfunc)
        print('Date successful: ' + str(success))
    
    output['mismatch index'] = M_t

    return output


def ucounterfactual(u,uopt,Mfunc):
    return

