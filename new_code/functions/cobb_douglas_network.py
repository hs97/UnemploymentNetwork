import numpy as np
import copy
from functions.helpers_price_quantities import pricing, wages, sectoral_output, aggregate_real_output
from functions.helpers_labor_market import theta, elasticity_labor_demand

class cobb_douglas_network:
    def __init__(self, Omega, elasticity_Dc, elasticity_fN, elasticity_Qtheta, tau, curlyF, elasticity_wtheta, elasticity_wA):
        #Initializes production network. All variables assumed to be at initial equilibrium values.
        self.Omega = Omega # matrix of production elasticities to intermediate goods for network at initial equilibrium
        self.elasticity_Dc = elasticity_Dc # elasticity final goods production to intermediate goods at initial equilibrium
        self.elasticity_fN = elasticity_fN # elasticity of production to labor input at intitial equilibrium
        self.elasticity_Qtheta = elasticity_Qtheta # elasticity of vacacy filling rate to tightness at initial equilibrium
        self.tau = tau # recruiter producer ratio at initial equilibrium
        self.curlyF = curlyF # recruiter producer ratio at initial equilibrium
        self.elasticity_wtheta = elasticity_wtheta # elasticity of wages to tightness at initial equilibrium
        self.elasticity_wA= elasticity_wA# elasticity of wages to productivity at initial equilibrium

        # Following variables are pinned down by the restrictions implied by Cobb-Douglas technology
        self.elasticity_lambdatheta = np.zeros_like(self.elasticity_wtheta) #elasticity of Domar weights to tightness
        self.elasticity_lambdaA = np.zeros_like(self.elasticity_wA) #elasticity of Domar weights to technology
        self.elasticity_elasticityfNtheta = np.zeros_like(self.elasticity_wtheta) # elasticity of elasticity of production to labor input to tightness
        self.elasticity_elasticityfNA= np.zeros_like(self.elasticity_wA) # elasticity of elasticity of production to labor input to productivity
   
        self.dloglambda = np.zeros_like(elasticity_Dc) # Domar weights are fixed
        self.dlogelasticity_fN = np.zeros_like(elasticity_Dc) # Sector level production elasticities are fixed
        self.dlogelasticity_Dc = np.zeros_like(elasticity_Dc) # Final production elasticities are fixed

        # Leontief inverse
        self.Nsector = self.Omega.shape[0]
        self.Psi = np.linalg.inv(np.eye(self.Nsector) - self.Omega)
        

    def shocks(self,dlogA,dlogH,H,L,U):
        # Calculates response of prices, quantities, labor market to shocks to productivity (dlogA) or the size of the labor force (dlogH)
        self.dlogA = dlogA
        self.dlogH = dlogH
        self.H = H
        self.L = L
        self.U = U
        
        self.elasticity_Ldtheta = elasticity_labor_demand(self.elasticity_fN, self.Psi, self.elasticity_wtheta, self.elasticity_lambdatheta,self.elasticity_elasticityfNtheta) 
        self.elasticity_LdA = elasticity_labor_demand(self.elasticity_fN, self.Psi, self.elasticity_wA, self.elasticity_lambdaA,self.elasticity_elasticityfNA) 
        
        # Tightness and prices
        self.dlogtheta = theta(self.dlogA, self.dlogH, self.curlyF, self.elasticity_Ldtheta, self.elasticity_LdA)
        self.dlogw = wages(self.dlogtheta,self.dlogA,self.elasticity_wtheta, self.elasticity_wA)
        self.dlogp = pricing(self.dlogw, self.dlogtheta, self.dlogA, self.elasticity_fN, self.elasticity_Qtheta, self.tau, self.Psi)
        
        # Output
        self.dlogy = sectoral_output(self.dlogp,self.dloglambda,self.dlogelasticity_fN,self.elasticity_fN,self.Psi)
        self.dlogY = aggregate_real_output(self.dlogp,self.dlogelasticity_fN,self.dlogelasticity_Dc,self.elasticity_Dc,self.elasticity_fN,self.Psi)

        # Unemployment
        self.dlogL = np.diag(self.curlyF.reshape(self.curlyF.shape[0],)) @ self.dlogtheta + self.dlogH
        self.dlogU = np.diag(np.power(U,-1).reshape(U.shape[0],)) @ (np.diag(H.reshape(H.shape[0],)) @ self.dlogH - np.diag(L.reshape(L.shape[0],)) @ self.dlogL)
        self.dUrate = (self.U + self.U * self.dlogU) / (self.H + self.H * self.dlogH) - self.U / self.H

        # Aggregate unemployment
        self.dlogLagg = self.L.T @ self.dlogL / np.sum(self.L)
        self.dlogHagg = self.H.T @ self.dlogH / np.sum(self.H)
        self.dlogUagg = self.U.T @ self.dlogU / np.sum(self.U)
        self.dUrate_agg = (np.sum(self.U) + np.sum(self.U) * self.dlogUagg) / (np.sum(self.H) + np.sum(self.H) * self.dlogHagg) - np.sum(self.U) / np.sum(self.H)

        return copy.deepcopy(self)

        
        

    def wage_elasticities(self,elasticity_wtheta, elasticity_wA):
        # Allows quick changes to wage elasticities
        self.elasticity_wtheta, self.elasticity_wA = elasticity_wtheta, elasticity_wA
        return copy.deepcopy(self)
