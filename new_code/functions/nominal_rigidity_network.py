import numpy as np
# Functions for multi-occupation network models with nominally rigid wages: dlog_w = 0


# output
def OutputFunc(dlog_A, dlog_H, dlog_K, Psi, curlyQ, curlyF, curlyT, curlyL, epsN, epsK, num=0):
    """ This function computes changes in sectoral output
        dlog_A: Jx1 tech shocks
        dlog_H: Ox1 labor force shocks
        dlog_K: Kx1 fixed factor shocks
        dlog_theta: Ox1 log tightness changes
        Psi: JxJ Leontief inverse
        curlyF: OxO elasticty of job finding wrt to theta
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        curlyT: OxO recruiter producer ratio
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production changes
        num: numeraire sector index
    Returns:
        dlog_y : Jx1 log output changes
    """
    Xi_theta = curlyL@Psi@epsN@(curlyF + curlyQ@curlyT)
    inv_mat = np.linalg.inv(curlyF - Xi_theta)
    Xi_nom = np.eye(dlog_H.shape[0]) + inv_mat@curlyL@Psi@epsN@curlyQ@curlyT
    Xi_nom = np.linalg.inv(Xi_nom) @ inv_mat

    num_mat = np.ones_like(epsK.T)
    num_mat[:,num] = 1
    Xi_y = np.eye(dlog_A.shape[0]) - Psi@epsN@(curlyF+curlyQ@curlyT)@Xi_nom@curlyL@Psi@epsK@num_mat
    Xi_y = np.linalg.inv(Xi_y)

    #coefficients
    cA = Psi
    cH = Psi@epsN@(np.eye(dlog_H.shape[0])+(curlyF + curlyQ@curlyT)@Xi_nom@(curlyL@Psi@epsN-np.eye(dlog_H.shape[0])))
    cK = Psi@epsK

    # changes in output
    dlog_y = Xi_y@(cA @ dlog_A + cH @ dlog_H + cK @ dlog_K)
    return dlog_y

def ThetaFunc(dlog_H, dlog_y, Psi, epsN, curlyL, curlyQ, curlyT, curlyF, num=0):
    """ This function computes changes in sectoral output
        dlog_H: Jx1 tech shocks
        dlog_y : Jx1 log output changes
        dlog_theta: Ox1 log tightness changes
        Psi: JxJ Leontief inverse
        curlyF: OxO elasticty of job finding wrt to theta
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        curlyT: OxO recruiter producer ratio
        curlyL: OxJ occupation-sector employment shares
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production changes
        num: numeraire sector index
    Returns:
        dlog_theta : Ox1 log tightness changes
    """
    Xi_theta = curlyL@Psi@epsN@(curlyF + curlyQ@curlyT)
    inv_mat = np.linalg.inv(curlyF - Xi_theta)
    Xi_nom = np.eye(dlog_H.shape[0]) + inv_mat@curlyL@Psi@epsN@curlyQ@curlyT
    Xi_nom = np.linalg.inv(Xi_nom) @ inv_mat
    num_mat = np.ones_like(epsN.T)
    num_mat[:,num] = 1

    #coefficients
    cy = Xi_nom@curlyL@Psi@epsN@num_mat
    cH = Xi_nom@(curlyL@Psi@epsN-np.eye(dlog_H.shape[0]))

    #changes in tightness
    dlog_theta = cy@dlog_y + cH@dlog_H

    return dlog_theta

#factor prices
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
    dlog_r = dlog_y_num - dlog_K
    return dlog_r

#sector prices
def PriceFunc(dlog_A, dlog_r, dlog_theta, Psi, epsN, epsK, curlyQ, curlyT):
    """ This function computes the vector of price responses
    Args:
        dlog_A : Jx1 tech shocks
        dlog_r: Kx1 change in fixed factor prices
        dlog_theta - Ox1 log tightness changes
        Psi: JxJ Leontief inverse
        curlyQ: OxO elasticity of vacancy filling wrt to theta
        epsN: JxO labor elasticity of production
        epsK: JxK fixed factor elasticity of production
        curlyT: OxO recruiter producer ratio
        num: indicates which price is numeraire
    """
    #coefficients
    cA = -Psi
    cr = Psi@epsK
    ct = -Psi@epsN@curlyQ@curlyT

    #changes in prices
    dlog_p = cA@dlog_A + cr@dlog_r + ct@dlog_theta
    return dlog_p