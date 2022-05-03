#File containing code for simple examples of labor markets 
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class labor_market:
    def __init__(self, lambdai, lambdaj, alphai=0.6, betai=0, Ai=1, wi=0.6,
                 kappai=0.84, si=0.035, etai=0.7,
                 omegai=0.67, Hi=1, alphaj=0.6, Aj=1, wj=0.6,
                 kappaj=0.84, sj=0.035, etaj=0.7, omegaj=0.67, 
                 Hj=1, yj = 1):
        #production function parameters: currently only setup for fully rigid wages
        self.alphai, self.betai, self.Ai, self.wi = alphai, betai, Ai, wi
        self.alphaj, self.Aj, self.wj, self.yj = alphaj, Aj, wj, yj
        #matching function parameters
        self.etai, self.omegai = etai, omegai
        self.etaj, self.omegaj = etaj, omegaj
        #labor market parameters
        self.kappai, self.si, self.Hi = kappai, si, Hi
        self.kappaj, self.sj, self.Hj = kappaj, sj, Hj
        # Unemployment search in other sectors
        self.lambdai, self.lambdaj = lambdai, lambdaj

    def job_finding(self, omega, theta, eta):
        return omega * theta ** (1 + eta )

    def vacancy_filling(self, omega, theta, eta):
        return omega * theta ** (-eta)
        
    def theta_exog(self, thetai, thetaj):
        self.thetai = thetai
        self.thetaj = thetaj
        
    def recruiter_producer(self, theta, omega, eta, kappa, s):
        q = self.vacancy_filling(omega,theta,eta)
        return  kappa * s / ( q - kappa * s )

    def labor_demand(self):
        self.taui = self.recruiter_producer(self.thetai, self.omegai, self.etai, self.kappai, self.si)
        self.tauj = self.recruiter_producer(self.thetaj, self.omegaj, self.etaj, self.kappaj, self.sj)
        own   = (self.alphai*self.Ai/(self.wi*(1 + self.taui)**self.alphai))**(1/(1 - self.alphai))
        other = (self.alphaj*(self.Aj**(1/self.alphaj))/(self.wj*(1 + self.tauj)))**(self.alphaj/(1 - self.alphaj)*self.betai/(1 - self.alphai)) 
        self.Ld = other * own
        return self.Ld
    
    def labor_supply(self, Lj):
        f = self.job_finding(self.omegai, self.thetai, self.etai)
        self.Ls = f/(self.si + self.lambdai*f)*(self.lambdai*self.Hi + (1 - self.lambdaj)*(self.Hj - Lj))
        return self.Ls

    def plot_Ld(self,lower=0.0001 ,upper=4):
        thetag = np.linspace(lower,upper,10000)
        self.theta_exog(thetag,thetag)
        Ld = self.labor_demand()
        plt.plot(Ld, thetag,'-b')
        plt.show()
        
    def plot_Ls(self, lower=0.0001, upper=4):
        thetag = np.linspace(lower, upper, 10000)
        self.theta_exog(thetag, thetag)
        Ls = self.labor_supply(lambdaj=0, Lj=5)
        plt.plot(Ls,thetag,'-r')
        plt.show()
        
    def plot_Ls_Ld(self,lower=0.0001,upper=4):
        thetag = np.linspace(lower,upper,10000)
        self.theta_exog(thetag,thetag)
        Ld = self.labor_demand()
        Ls = self.labor_supply()
        
        plt.plot(Ld,thetag,'-b')
        plt.plot(Ls,thetag,'-r')
        plt.axvline(x=self.Hi)
        plt.show()

    def production_function(self):
        self.yi = self.Ai * (self.Ld / (1 + self.taui) ) ** self.alphai * self.yj ** self.betai
        return self.yi

class economy:
    def __init__(self, market1, market2):
        self.market1 = market1
        self.market2 = market2

    def equilibrium_condition(self, theta_vec):
        self.market1.thetai, self.market1.thetaj = theta_vec[0], theta_vec[1]
        self.market2.thetai, self.market2.thetaj = theta_vec[1], theta_vec[0]

        self.market1.labor_demand()
        self.market2.labor_demand()
        self.market1.labor_supply(self.market2.Ld)
        self.market2.labor_supply(self.market1.Ld)
            
        obj_market1 = np.abs(self.market1.Ld - self.market1.Ls)
        obj_market2 = np.abs(self.market2.Ld - self.market2.Ls)

        return np.array([obj_market1, obj_market2])

    def solve_equilibrium(self, theta_guess=np.array([])):
        if theta_guess.shape[0]==0:
            self.theta_guess = np.zeros((2,))
        else:
            self.theta_guess = theta_guess
        self.equilibrium_solver_output = opt.root(self.equilibrium_condition,
                                                  self.theta_guess, method='broyden2')
        self.theta_star = self.equilibrium_solver_output.x
        
        self.market1.thetai, self.market1.thetaj = self.theta_star[0], self.theta_star[1]
        self.market2.thetai, self.market2.thetaj = self.theta_star[1], self.theta_star[0]
        self.market1.labor_demand()
        self.market2.labor_demand()
        self.market1.labor_supply(self.market2.Ld)
        self.market2.labor_supply(self.market1.Ld)
        self.market1.production_function()
        self.market2.production_function()
        U1 = self.market1.lambdai*(self.market1.Hi - self.market1.Ld) + (1 - self.market1.lambdaj)*(self.market2.Hi - self.market2.Ld)
        U2 = self.market2.lambdai*(self.market2.Hi - self.market2.Ld) + (1 - self.market2.lambdaj)*(self.market1.Hi - self.market1.Ld)
        self.market1.urate = U1/(U1 + self.market1.Ld)
        self.market2.urate = U2/(U2 + self.market2.Ld)

        #self.market1.urate = (self.market1.Hi - self.market1.Ld)/self.market1.Hi
        #self.market2.urate = (self.market2.Hi - self.market2.Ld)/self.market2.Hi
        print(f"Equilibrium labor demand and supply for market 1 are {self.market1.Ld, self.market1.Ls}")     
        print(f"Equilibrium labor demand and supply for market 2 are {self.market2.Ld, self.market2.Ls}")     
        return self.theta_star
    

#upstream = labor_market(lambdai=1, lambdaj=1, betai=0)
#downstream = labor_market(lambdai=1, lambdaj=1, betai=0.8)
#immobile_labor = economy(upstream,downstream)
#immobile_labor.solve_equilibrium()

#upstream = labor_market(lambdai=0.5, lambdaj=0.5, betai=0)
##downstream = labor_market(lambdai=0.5, lambdaj=0.5, betai=0.8)
#mobile_labor = economy(upstream,downstream)
#mobile_labor.solve_equilibrium()

#upstream = labor_market(lambdai=0.75, lambdaj=0.75, betai=0)
#downstream = labor_market(lambdai=0.75, lambdaj=0.75, betai=0.8)
#partial_mobile_labor = economy(upstream,downstream)
#partial_mobile_labor.solve_equilibrium()



