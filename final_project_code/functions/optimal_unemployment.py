import numpy as np
from numba import njit
import pandas as pd

# cobb-douglas matching function and first derivative

def m_cd(u,v,θ,η):
    # takes np arrays ofvacancies, unemployment rates, matching efficiencies, and vacancy weight in matching function
    # returns number of matches
    U = np.power(u,1-η)
    V = np.power(v,η)
    return θ*V*U

def mu_cd(u,v,θ,η):
    # takes np arrays ofvacancies, unemployment rates, matching efficiencies, and vacancy weight in matching function
    # returns marginal increase matches per increase in unemployment
    τ = np.power(v/u,η)
    return θ*τ

def Lstar(u,v,e,θ,η,m):
    # takes np arrays ofvacancies, unemployment rates, existing labor stocks, matching efficiencies, parameters of matching function, and a matching function
    # in cobb-douglas case, η is just the vacancy weight of vacancies, but can accomodate more general cases
    # realized labor in each sector
    return e + m(u,v,θ,η)

def objective(u,v,θ,η,λ,α):
    return
