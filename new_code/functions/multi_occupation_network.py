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
def ThetaFunc(dlog_A, dlog_H, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN):
    # dlog_A     - Jx1 tech shocks
    # dlog_H     - Ox1 labor force shocks
    # dlog_wR    - Ox1 log adjusted wage changes
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
    O = dlog_H.shape[0]
    
    # Creating matrices
    Xi = curlyL @ Psi @ epsN @ (curlyF + curlyQ @ curlyT)
    inv_mat = np.linalg.inv(curlyF - Xi)
    I = np.eye(O)
    
    # Contribution of different components
    Cw = -inv_mat 
    Ca = inv_mat @ curlyL @ Psi
    Ch = inv_mat @ (curlyL @ Psi @ epsN - I)
    Ce = -inv_mat @ curlyL @ Psi
    Cλ = inv_mat @ curlyL @ Psi @ (np.diag(np.sum(Omega,1)) - Omega)

    # Change in tightness
    dlog_theta = Cw @ dlog_wR + Ch @ dlog_H + inv_mat @ np.diag(curlyL @ dlog_epsN).reshape((O,1)) + Cλ @ dlog_lam + Ce @ curlyE + Ca @ dlog_A
    
    return dlog_theta

# Price changes
def PriceFunc(dlog_A, dlog_wR, dlog_theta, Psi, curlyQ, epsN, curlyT, curlyL, num=0):
    # dlog_A     - Jx1 tech shocks
    # dlog_wR    - Ox1 log adjusted wage changes
    # dlog_theta - Ox1 log tightness changes
    # Psi        - JxJ Leontief inverse
    # curlyQ     - OxO elasticity of vacancy filling wrt to theta
    # epsN       - JxO labor elasticity of production
    # curlyT     - OxO recruiter producer ratio
    # dlog_p     - Jx1 log price changes
    # num        - indicates which price is numeraire

    # Contributions of different components
    Cw = Psi @ epsN
    Ctheta = -Cw @ curlyQ @ curlyT
    Ca = -Psi
    # Imposing numeraire
    Cw[num, :] = 0
    Ctheta[num, :] = 0
    Ca[num, :] = 0
    # Inverse matrix
    Xi = np.eye(Psi.shape[0]) - Psi @ epsN @ curlyL
    Xi[num, :] = 0
    Xi[num, num] = 1
    inv_mat = np.linalg.inv(Xi)

    # Price changes
    dlog_p = inv_mat @ (Cw @ dlog_wR + Ctheta @ dlog_theta + Ca @ dlog_A)
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
def LaborDemand(dlog_wR, dlog_y, dlog_p, dlog_epsN, curlyL):
    # dlog_wr    - Ox1 log adjusted wage changes
    # dlog_y     - Jx1 log output changes
    # dlog_p     - Jx1 log price changes
    # dlog_epsN  - JxO labor elasticity of production changes
    # curlyL     - OxJ occupation-sector employment shares

    O = dlog_wR.shape[0]
    dlog_Ld = curlyL @ dlog_y + np.diag(curlyL @ dlog_epsN).reshape((O,1)) - dlog_wR
    return dlog_Ld

# Aggregate output
def AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD):
    # dlog_y     - Jx1 log output changes
    # dlog_lam   - Jx1 Domar weight changes
    # dlog_epsD  - Jx1 log changes in consumption elasticities
    # epsD       - Jx1 consumption elasticities

    dlog_aggY = epsD.T @ (dlog_epsD + dlog_y - dlog_lam)
    return dlog_aggY

def WageElasticityFunc(gamma_A, gamma_H, Psi, curlyL, epsN):
    # gamma_A    - adjust responsiveness to technology
    # gamma_H    - adjust responsiveness to labor force
    # Psi        - JxJ Leontief inverse
    # curlyL     - OxJ occupation-sector employment shares
    # epsN       - JxO labor elasticity of production

    epsW_A = gamma_A * curlyL @ Psi
    epsW_H = gamma_H * (curlyL @ Psi @ epsN - np.eye(curlyL.shape[0]))
    return epsW_A, epsW_H

def UnemploymentFunc(dlog_L, dlog_H):
    # dlog_L     - Ox1 change in labor (demand or supply)
    # dlog_H     - Ox1 labor force shocks
    dlog_U = dlog_H - dlog_L
    return dlog_U