import numpy as np
# Functions for multi-occupation network models

# Generic wage schedule
def WageFunc(dlog_A, dlog_H, epsW_A, epsW_H):
    # dlog_A - Jx1 tech shocks
    # dlog_H - Ox1 labor force shocks
    # epsW_A - OxJ elasticity of wages to A
    # epsW_H - OxO elasticity of wages to H
    # dlog_w - Ox1 log wage changes
    dlog_w = epsW_A @ dlog_A + epsW_H @ dlog_H
    return dlog_w

# Creating curlyE 
def curlyEFunc(dlog_epsN,epsN):
    # dlog_epsN - JxO changes in production elasticities wrt labor types
    # epsN      - JxO production elasticities wrt labor types
    # curlyE    - Jx1 reweighted log labor type elasticity changes
    J = epsN.shape[0]
    curlyE = np.diag(epsN @ dlog_epsN.T).reshape(J,1)
    return curlyE

# Tightness changes
def ThetaFunc(dlog_A, dlog_H, dlog_w, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, num=0):
    # dlog_A     - Jx1 tech shocks
    # dlog_H     - Ox1 labor force shocks
    # dlog_w     - Ox1 log wage changes
    # dlog_epsN  - JxO labor elasticity of production changes
    # dlog_lam   - Jx1 Domar weight changes
    # Psi        - JxJ Leontief inverse
    # Omega      - JxJ elasticity of production wrt to intermediate inputs
    # curlyF     - OxO elasticty of job finding wrt to theta
    # curlyQ     - OxO elasticity of vacancy filling wrt to theta
    # curlyT     - OxO recruiter producer ratio
    # curlyE     - JxJ reweigted labor elasticity of production changes
    # curlyL     - OxJ occupation-sector employment shares
    # epsN       - JxO labor elasticity of production
    # dlog_theta - Ox1 log changes in tightness
    J = dlog_A.shape[0]
    O = dlog_H.shape[0]
    
    # Creating matrices
    curlyL_j = np.zeros_like(curlyL)
    curlyL_j[:,num] = curlyL[:,num]
    Xi = curlyL @ Psi @ epsN @ curlyF + curlyL_j @ Psi @ epsN @ curlyQ @ curlyT 
    inv_mat = np.linalg.inv(curlyF - Xi)
    I = np.eye(O)
    
    # Contribution of different components
    Cw = inv_mat @ ((curlyL-curlyL_j) @ Psi @ epsN - np.diag(np.sum(curlyL,1)))
    Ca = inv_mat @ curlyL_j @ Psi
    Ch = inv_mat @ (curlyL @ Psi @ epsN - I)
    Ce = -inv_mat @ curlyL @ Psi
    Cλ = inv_mat @ curlyL @ Psi @ (np.diag(np.sum(Omega,1)) - Omega)

    # Change in tightness
    dlog_theta = Cw @ dlog_w + Ch @ dlog_H + inv_mat @ np.diag(curlyL @ dlog_epsN).reshape((O,1)) + Cλ @ dlog_lam + Ce @ curlyE + Ca @ dlog_A
    
    return dlog_theta

# Price changes
def PriceFunc(dlog_A, dlog_w, dlog_theta, Psi, curlyQ, epsN, curlyT):
    # dlog_A     - Jx1 tech shocks
    # dlog_w     - Ox1 log wage changes
    # dlog_theta - Ox1 log tightness changes
    # Psi        - JxJ Leontief inverse
    # curlyQ     - OxO elasticity of vacancy filling wrt to theta
    # epsN       - JxO labor elasticity of production
    # curlyT     - OxO recruiter producer ratio
    # dlog_p     - Jx1 log price changes

    # Contributions of different components
    Cw = Psi @ epsN
    Ctheta = -Cw @ curlyQ @ curlyT
    Ca = -Psi

    # Price changes
    dlog_p = Cw @ dlog_w + Ctheta @ dlog_theta + Ca @ dlog_A
    return dlog_p

# Output changes
def OutputFunc(dlog_A, dlog_H, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, curlyT, curlyE):
    # dlog_A     - Jx1 tech shocks
    # dlog_H     - Ox1 labor force shocks
    # dlog_theta - Ox1 log tightness changes
    # dlog_lam   - Jx1 Domar weight changes
    # Psi        - JxJ Leontief inverse
    # Omega      - JxJ elasticity of production wrt to intermediate inputs
    # curlyQ     - OxO elasticity of vacancy filling wrt to theta
    # curlyF     - OxO elasticty of job finding wrt to theta
    # epsN       - JxO labor elasticity of production
    # curlyT     - OxO recruiter producer ratio
    # curlyE     - JxJ reweigted labor elasticity of production changes
    # dlog_y     - Jx1 log output changes

    # Contributions of different coponents
    Ca = Psi
    Ch = Psi @ epsN
    Ctheta = Ch @ (curlyF + curlyQ @ curlyT)
    Ce = -Psi
    Cλ = Psi @ (np.diag(np.sum(Omega,1)) - Omega)
    
    # Output changes
    dlog_y = Ca @ dlog_A + Ctheta @ dlog_theta + Ch @ dlog_H + Ce @ curlyE + Cλ @ dlog_lam

    return dlog_y

# Labor supply
def LaborSupply(dlog_H,dlog_theta,curlyF):
    # dlog_H     - Ox1 labor force shocks
    # dlog_theta - Ox1 log tightness changes
    # curlyF     - OxO elasticty of job finding wrt to theta

    dlog_Ls = curlyF @ dlog_theta + dlog_H
    return dlog_Ls

# Labor demand    
def LaborDemand(dlog_w, dlog_y, dlog_p, dlog_epsN, curlyL):
    # dlog_w     - Ox1 log wage changes
    # dlog_y     - Jx1 log output changes
    # dlog_p     - Jx1 log price changes
    # dlog_epsN  - JxO labor elasticity of production changes
    # curlyL     - OxJ occupation-sector employment shares

    O = dlog_w.shape[0]
    dlog_Ld = curlyL @ (dlog_p + dlog_y) + np.diag(curlyL @ dlog_epsN).reshape((O,1)) - dlog_w
    return dlog_Ld

# Aggregate output
def AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD):
    # dlog_y     - Jx1 log output changes
    # dlog_lam   - Jx1 Domar weight changes
    # dlog_epsD  - Jx1 log changes in consumption elasticities
    # epsD       - Jx1 consumption elasticities

    dlog_aggY = epsD.T @ (dlog_epsD + dlog_y - dlog_lam)
    return dlog_aggY