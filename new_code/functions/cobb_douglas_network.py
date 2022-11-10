import numpy as np
import copy
from functions.helpers_price_quantities import pricing, wages, sectoral_output, aggregate_real_output
from functions.helpers_labor_market import theta, elasticity_labor_demand
import matplotlib.pyplot as plt

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
        self.dlog_relative_wages = self.dlogw-self.dlogp

        # Output
        self.dlogy = sectoral_output(self.dlogp,self.dloglambda,self.dlogelasticity_fN,self.elasticity_fN,self.Psi)
        self.dlogY = aggregate_real_output(self.dlogp, self.dlogelasticity_fN, self.dlogelasticity_Dc, self.elasticity_Dc, self.elasticity_fN,self.Psi)

        # Unemployment
        self.dlogL = np.diag(self.curlyF.reshape(self.curlyF.shape[0],)) @ self.dlogtheta + self.dlogH
        self.dlogU = np.diag(np.power(U,-1).reshape(U.shape[0],)) @ (np.diag(H.reshape(H.shape[0],)) @ self.dlogH - np.diag(L.reshape(L.shape[0],)) @ self.dlogL)
        self.dUrate = (self.U + self.U * self.dlogU) / (self.H + self.H * self.dlogH) - self.U / self.H

        # Aggregate unemployment
        self.dlogLagg = self.L.T @ self.dlogL / np.sum(self.L)
        self.dlogHagg = self.H.T @ self.dlogH / np.sum(self.H)
        self.dlogUagg = self.U.T @ self.dlogU / np.sum(self.U)
        self.dUrate_agg = (np.sum(self.U) + np.sum(self.U) * self.dlogUagg) / (np.sum(self.H) + np.sum(self.H) * self.dlogHagg) - np.sum(self.U) / np.sum(self.H)

        self.output_dict = {'dlogA':self.dlogA, 'dlogH':self.dlogH, 'dlogtheta':self.dlogtheta, 'dlogw':self.dlogw, 'dlogp':self.dlogp, 'dlog_relative_wages':self.dlog_relative_wages, 'dlogy':self.dlogy, 'dlogY':self.dlogY, 'dlogL':self.dlogL, 'dlogU':self.dlogU, 'dUrate':self.dUrate, 'dlogLagg':self.dlogLagg, 'dlogHagg':self.dlogHagg, 'dlogUagg':self.dlogUagg, 'dUrate_agg':self.dUrate_agg}

        return copy.deepcopy(self)

        
        

    def wage_elasticities(self,elasticity_wtheta, elasticity_wA):
        # Allows quick changes to wage elasticities
        self.elasticity_wtheta, self.elasticity_wA = elasticity_wtheta, elasticity_wA
        return copy.deepcopy(self)


def bar_plot(networks, sector_names, varname, aggname, xlab, ylab, labels, save_path, fontsize=15, barWidth = 0.25, dpi=300):
    fig = plt.subplots(dpi=dpi)

    #creating arrays
    br = np.zeros((len(sector_names),len(networks)))
    yvals = np.zeros((len(sector_names)+1,len(networks)))
    nsector = networks[0].output_dict[varname].shape[0]

    #initializing
    br[0,:] = np.arange(len(sector_names))
    yvals[0,:nsector] = networks[0].output_dict[varname]
    if len(sector_names)>nsector:
        yvals[0,-1] = networks[0].output_dict[aggname]
    plt.bar(br[0,:], yvals[0,:], width = barWidth,
        edgecolor ='grey', label =labels[0])
    
    #looping through other networks
    for i in range(1,len(networks)):
        br[i,:] = [x + barWidth for x in br[i-1,:]]
        yvals[i,:nsector] = networks[i].output_dict[varname]
        if len(sector_names)>nsector:
            yvals[i,-1] = networks[i].output_dict[aggname]
        plt.bar(br[i,:], yvals[i,:], width = barWidth,
            edgecolor ='grey', label =labels[i])
    
    plt.xlabel(xlab, fontweight ='bold', fontsize = fontsize)
    plt.ylabel(ylab, fontweight ='bold', fontsize = fontsize)
    plt.xticks([r + barWidth for r in range(len(sector_names))],
        sector_names)
    plt.savefig(save_path,dpi=dpi)    
