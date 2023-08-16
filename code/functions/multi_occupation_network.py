import numpy as np
# Functions for multi-occupation network models

# Generic wage schedule
def WageFunc(dlog_A, dlog_H, dlog_K, epsW_A, epsW_H, epsW_K):
    """
    This function computes wage responses given wage elasticities and the shocks
    Args: 
        dlog_A: Jx1 tech shocks
        dlog_H: Ox1 labor force shocks
        dlog_K: Kx1 fixed factor shocks
        epsW_A: OxJ elasticity of wages to A
        epsW_H: OxO elasticity of wages to H
        epsW_K: OxK elasticity of wages to K
    Returns:
        dlog_wR: Ox1 log adjusted wage changes
    """

    dlog_wR = epsW_A@dlog_A + epsW_H@dlog_H + epsW_K@dlog_K
    return dlog_wR

# Creating curlyE 
def curlyEFunc(dlog_epsN,epsN):
    """This function computes curly E
    Args: 
        dlog_epsN: JxO changes in production elasticities wrt labor types
        epsN: JxO production elasticities wrt labor types
    Returns:
        curlyE: Jx1 reweighted log labor type elasticity changes
    """

    J = epsN.shape[0]
    curlyE = np.diag(epsN@dlog_epsN.T).reshape(J,1)
    return curlyE

# Tightness changes
def ThetaFunc(dlog_A, dlog_H, dlog_K, dlog_wR, dlog_epsN, dlog_lam, Psi, Omega, curlyF, curlyQ, curlyT, curlyE, curlyL, epsN, epsK):
    """ This function computes changes in tightness

    TODO: This function has not yet been updated to accomodate general CRTS tightness after incorporating fixed factors
    Args: 
        dlog_A: Jx1 tech shocks
        dlog_H: Ox1 labor force shocks
        dlog_K: Kx1 fixed factor shocks
        dlog_wR: Ox1 log adjusted wage changes
        dlog_epsN: JxO labor elasticity of production changes
        dlog_lam: Jx1 Domar weight changes
        Psi: JxJ Leontief inverse
        Omega: JxJ elasticity of production wrt to intermediate inputs
        curlyF: OxO elasticty of job finding wrt to theta
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        curlyT: OxO recruiter producer ratio
        curlyE: JxJ reweigted labor elasticity of production changes
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production changes
    Returns:
        dlog_theta - Ox1 log changes in tightness
    """

    O = dlog_H.shape[0]
    
    # Creating matrices
    Xi = curlyL@Psi@epsN@(curlyF + curlyQ@curlyT)
    inv_mat = np.linalg.inv(curlyF - Xi)
    I = np.eye(O)
    
    # Contribution of different components
    Cw = -inv_mat 
    Ca = inv_mat@curlyL@Psi
    Ch = inv_mat@(curlyL@Psi@epsN - I)
    Ck = inv_mat@curlyL@Psi@epsK
    Ce = -inv_mat@curlyL@Psi
    Cλ = inv_mat@curlyL@Psi@(np.diag(np.sum(Omega,1)) - Omega)

    # Change in tightness
    dlog_theta = Cw@dlog_wR + Ch@dlog_H + Ck@dlog_K + inv_mat@np.diag(curlyL@dlog_epsN).reshape((O,1)) + Cλ@dlog_lam + Ce@curlyE + Ca@dlog_A
    return dlog_theta

# Price changes
def PriceFunc(dlog_A, dlog_r, dlog_wR, dlog_theta, Psi, curlyQ, epsN, epsK, curlyT, curlyL, num=0):
    """ This function computes the vector of price responses
    Args:
        dlog_A : Jx1 tech shocks
        dlog_r: Kx1 change in fixed factor prices
        dlog_wR: Ox1 log adjusted wage changes
        dlog_theta - Ox1 log tightness changes
        Psi: JxJ Leontief inverse
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production
        curlyT: OxO recruiter producer ratio
        curlyL: OxJ occupation-sector employment shares
        dlog_p: Jx1 log price changes
        num: indicates which price is numeraire
    """
    # Contributions of different components
    Cw = Psi@epsN
    Ctheta = -Cw@curlyQ@curlyT
    Ca = -Psi
    Cr = Psi@epsK
    # Imposing numeraire
    Cw[num, :] = 0
    Ctheta[num, :] = 0
    Ca[num, :] = 0
    Cr[num, :] = 0

    # Inverse matrix
    Xi = np.eye(Psi.shape[0]) - Psi@epsN@curlyL
    Xi[num, :] = 0
    Xi[num, num] = 1
    inv_mat = np.linalg.inv(Xi)

    # Price changes
    dlog_p = inv_mat@(Cw@dlog_wR + Ctheta@dlog_theta + Ca@dlog_A + Cr@dlog_r)
    return dlog_p

def rFunc(dlog_y, dlog_K, num):
    """ This function computes dlog_r
    Args:
        dlog_y: Jx1 matrix of sectoral output changes
        dlog_K: Kx1 matrix of fixed factor shocks
        num: numeraire
    Returns:
        dlog_r: Kx1 matrix of fixed fator price changes
    """
    dlog_y_num = dlog_y[num]
    return dlog_y_num - dlog_K

# Output changes
def OutputFunc(dlog_A, dlog_H, dlog_K, dlog_theta, dlog_lam, Psi, Omega, curlyQ, curlyF, epsN, epsK, curlyT, curlyE):
    """ This function computes changes in sectoral output

    TODO: This function has not yet been updated to accomodate general CRTS tightness after incorporating fixed factors
    Args: 
        dlog_A: Jx1 tech shocks
        dlog_H: Ox1 labor force shocks
        dlog_K: Kx1 fixed factor shocks
        dlog_theta: Ox1 log tightness changes
        dlog_lam: Jx1 Domar weight changes
        Psi: JxJ Leontief inverse
        Omega: JxJ elasticity of production wrt to intermediate inputs
        curlyF: OxO elasticty of job finding wrt to theta
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        curlyT: OxO recruiter producer ratio
        curlyE: JxJ reweigted labor elasticity of production changes
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production changes
    Returns:
        dlog_y : Jx1 log output changes
    """

    # Contributions of different coponents
    Ca = Psi
    Ch = Psi@epsN
    Ck = Psi@epsK
    Ctheta = Ch@(curlyF + curlyQ@curlyT)
    Ce = -Psi
    Cλ = Psi@(np.diag(np.sum(Omega,1)) - Omega)
    
    # Output changes
    dlog_y = Ca@dlog_A + Ctheta@dlog_theta + Ch@dlog_H + Ck@dlog_K + Ce@curlyE + Cλ@dlog_lam

    return dlog_y

# Labor supply
def LaborSupply(dlog_H, dlog_theta, curlyF):
    # dlog_H : Ox1 labor force shocks
    # dlog_theta - Ox1 log tightness changes
    # curlyF : OxO elasticty of job finding wrt to theta

    dlog_Ls = curlyF@dlog_theta + dlog_H
    return dlog_Ls

# Labor demand    
def LaborDemand(dlog_wR, dlog_y, dlog_epsN, curlyL):
    # dlog_wr: Ox1 log adjusted wage changes
    # dlog_y : Jx1 log output changes
    # dlog_epsN: JxO labor elasticity of production changes
    # curlyL : OxJ occupation-sector employment shares

    O = dlog_wR.shape[0]
    dlog_Ld = curlyL @ dlog_y + np.diag(curlyL @ dlog_epsN).reshape((O,1)) - dlog_wR
    return dlog_Ld

# Aggregate output
def AggOutputFunc(dlog_y, dlog_lam, dlog_epsD, epsD):
    # dlog_y : Jx1 log output changes
    # dlog_lam : Jx1 Domar weight changes
    # dlog_epsD: Jx1 log changes in consumption elasticities
    # epsD   : Jx1 consumption elasticities

    dlog_aggY = epsD.T@(dlog_epsD + dlog_y - dlog_lam)
    return dlog_aggY

def WageElasticityFunc(gamma_A, gamma_H, gamma_K, Psi, curlyL, epsN, epsK):
    """
    This function computes the wage elasticities based on 
    gamma deviations from Hulten's theorem
    Args: 
        gamma_A: adjust responsiveness to technology
        gamma_H: adjust responsiveness to labor force
        Psi: JxJ Leontief inverse
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production
    Returns:
        epsW_A: OxJ elasticity of wages to A
        epsW_H: OxO elasticity of wages to H
        epsW_K: OxK elasticity of wages to K
    """
    epsW_A = gamma_A * curlyL @ Psi
    epsW_H = gamma_H * (curlyL @ Psi @ epsN - np.eye(curlyL.shape[0]))
    epsW_K = gamma_K * curlyL @ Psi @ epsK
    return epsW_A, epsW_H, epsW_K

def WageElasticityFuncMP(gamma, Psi, epsN, epsK, curlyF, curlyQ, curlyT, curlyL):
    """
    This function computes the wage elasticities based on gamma 
    deviations from changes in the marginal product of labor
    Args: 
        gamma: adjust responsiveness to marginal product of labor
        Psi: JxJ Leontief inverse
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production
        curlyQ: OxO elasticity of job filling rate to tightness
        curlyT: OxO recruiter producer ratio
    Returns:
        epsW_A: OxJ elasticity of wages to A
        epsW_H: OxO elasticity of wages to H
        epsW_K: OxK elasticity of wages to K
    """

    XiMP = gamma/(1-gamma) * curlyQ@curlyT + curlyL@Psi@epsN@(curlyF + curlyQ@curlyT)
    invTerm = np.linalg.inv(curlyF - XiMP)
    commonTerm = -1 * gamma/(1-gamma) * curlyQ@curlyT@invTerm

    epsW_A = commonTerm @ curlyL @ Psi
    epsW_H = commonTerm @ (curlyL @ Psi @ epsN - np.eye(curlyL.shape[0]))
    epsW_K = commonTerm @ curlyL @ Psi @ epsK

    return epsW_A, epsW_H, epsW_K


def UnemploymentRateFunc(dlog_theta, theta, curlyF, phi):
    # dlog_L : Ox1 change in labor (demand or supply)
    # dlog_H : Ox1 labor force shocks
    f = np.diag(np.diag(phi) * np.diag(theta)**(1-np.diag(curlyF)))
    dlog_u = - f @ np.linalg.inv((np.eye(f.shape[0])-f)) @ curlyF @ dlog_theta
    return dlog_u

def AggUnemploymentRateFunc(dlog_H, dlog_u, U, H):
    U = np.diag(U.flatten())
    H = np.diag(H.flatten())
    dlog_uagg = np.ones((1,U.shape[0]))@((U/np.sum(U)-H/np.sum(H)) @ dlog_H - (U/np.sum(U)) @ dlog_u) 
    return dlog_uagg

def AggThetaFunc(dlog_theta, dlog_U, U, V):
    dlog_V = dlog_theta + dlog_U 
    dlog_Uagg  = U.T @ dlog_U / np.sum(U)
    dlog_Vagg  = V.T @ dlog_V / np.sum(V)
    dlog_thetaAgg = dlog_Vagg - dlog_Uagg
    return dlog_thetaAgg