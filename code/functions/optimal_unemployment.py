import numpy as np
from numba import njit
import pandas as pd
from scipy.optimize import root
import matplotlib.pyplot as plt

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

def ustar_objective(uopt,param):
    
    v = param['v']
    e = param['e']
    φ = param['φ']
    η = param['η']
    λ = param['λ']
    α = param['α']
    mfunc  = param['mfunc']
    mufunc = param['mufunc']
    Lfunc  = param['Lfunc']
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


# Production function and social welfare function, check that solution improves welfare
def production_function(A,x,L,α):
    Xin = np.prod(np.power(x,A),axis=1)
    Lin = np.power(L,α)
    y   = Lin*Xin #(Nsector,) array  
    return y

def firm_FOC(A,x,y,p):
    # n^2 restrictions from firms FOC
    P   = np.tile(p,(p.shape[0],1))
    px  = P*x
    py  = np.tile(p*y,(p.shape[0],1)).T
    out = A - np.divide(px,py)
    return out.reshape((p.shape[0]**2,))

def household_FOC(θ,C,p):
    #n-1 restrictions from household FOC
    λ = np.divide(θ,p*C)
    out = λ[1:] - λ[0]
    return out

def market_clearing(y,C,x):
    # n restrictions from market clearing 
    out = y - C - np.sum(x,axis=0)
    return out

def full_solution_objective(opt,param):
    θ = param['θ']
    if np.any(opt <= 0):
        obj = np.ones((θ.shape[0]**2+2*θ.shape[0]-1,))*1e12
    else:
        α = param['α']
        A = param['A']
        L = param['L']
        yfunc    = param['yfunc']
        mkt_func = param['mkt_func']
        hh_foc   = param['hh_foc']
        f_foc    = param['f_foc']

        n = L.shape[0]
        x = opt[:n**2].reshape((n,n))
        C = opt[n**2:n**2+n]
        
        p = np.zeros_like(L)
        p[0]  = 1
        p[1:] = opt[n**2+n:]

        y = yfunc(A,x,L,α)

        obj = np.zeros_like(opt)
        obj[:n**2] = f_foc(A,x,y,p)
        obj[n**2:n**2+n-1] = hh_foc(θ,C,p)
        obj[n**2+n-1:] = mkt_func(y,C,x)
    return obj

def root_robust(objective,param,uguess_mean=np.array([]),tol=1e-6,maxiter=1e4,ntrue=100,guessrange=0.1):
    #wrapper for scipy root in ustar notation, also implements robustness to intial guess with randomly generated intital guesses
    if uguess_mean.shape[0] == 0:
        raise Exception('Must provide initial guess')
    count_true = 0
    out_mat    = np.array([])
    itercount = 0
    while count_true<ntrue and itercount<maxiter:
        uguess = np.zeros_like(uguess_mean)
        for i in range(uguess_mean.shape[0]):
            uguess[i] = np.random.uniform(uguess_mean[i]-guessrange/2,uguess_mean[i]+guessrange/2,1)
        uguess = np.abs(uguess)
        us = root(objective,uguess,args=(param),method='hybr',tol=tol)
        count_true += us.success
        itercount  += 1
        print('Num attempt: ' + str(itercount))
        if us.success == True:
            out_mat = np.append(out_mat,us.x)
            print('Num converged: ' + str(count_true))
    out_mat = out_mat.reshape((ntrue,uguess_mean.shape[0]))
    out_gap = out_mat - out_mat[0,:]
    if np.max(np.abs(out_gap))>10*tol:
        success = False
    else:
        success = True
    return  np.mean(out_mat,axis=0), success

# Mismatch index measures and optimal unemployment 
def Mindex(u,uopt,v,φ,η,mfunc):
    h    = mfunc(u,v,φ,η)
    hopt = mfunc(uopt,v,φ,η)
    return 1 - np.sum(h)/np.sum(hopt)

def Mindex_sectoral(u,uopt,v,φ,η,mfunc):
    h    = mfunc(u,v,φ,η)
    hopt = mfunc(uopt,v,φ,η)
    return 1 - h/hopt


# Function that runs code once for each time period in the data

class mismatch_estimation:
    def __init__(self,df,param,tol=1e-6,maxiter=1e5,ntrue=100,guessrange=0.1,outpath='code/output/'):
        self.input  = df
        self.input  = self.input.rename(columns={'v':'vraw','u':'uraw','e':'eraw'})
        self.input['v'] = self.input['vraw']
        self.input['u'] = self.input['uraw']
        self.input['e'] = self.input['eraw']
        
        self.param  = param
        self.param['Nsector']  = self.input.BEA_sector.unique().shape[0]
        self.param['outpath']  = outpath

        self.output = pd.DataFrame(index=df.date.unique(),columns=df.BEA_sector.unique())
        self.M_t    = np.zeros(df.date.unique().shape[0])
        for i in range(df.date.unique().shape[0]):
            vraw = np.array(self.input.vraw[self.input.date==self.input.date.unique()[i]])
            uraw = np.array(self.input.uraw[self.input.date==self.input.date.unique()[i]])
            eraw = np.array(self.input.eraw[self.input.date==self.input.date.unique()[i]])
            e    = eraw/np.sum(eraw+uraw)
            u    = uraw/np.sum(eraw+uraw)
            v    = vraw/np.sum(eraw+uraw)
            self.input.e.iloc[self.input.date==self.input.date.unique()[i]] = e 
            self.input.u.iloc[self.input.date==self.input.date.unique()[i]] = u 
            self.input.v.iloc[self.input.date==self.input.date.unique()[i]] = v 
            self.param['v'] = v
            self.param['e'] = e
            
            ustar_t, success = root_robust(self.param['objective'],self.param,uguess_mean=u,tol=tol,maxiter=maxiter,ntrue=ntrue,guessrange=guessrange)
            self.output.iloc[i,:] = ustar_t
            self.M_t[i]  = Mindex(u,ustar_t,v,self.param['φ'],self.param['η'],self.param['mfunc'])
            print('Starting date: ' + str(df.date.unique()[i]))
            print('Date successful: ' + str(success))
    
        self.output['mismatch_index'] = self.M_t 

    def mHP(self,HP_λ,fname,dpi):
        self.output['mismatch_trend'], self.output['mismatch_cycle'] = HP(self.M_t,HP_λ)
        self.mHP_plot, ax = plt.subplots(1,1,dpi = dpi)
        ax.plot(self.output.index,self.output.mismatch_trend,'-k')
        ax.set_xlabel('Date')
        ax.set_ylabel('Mismatch index')
        plt.savefig(self.param['outpath'] + fname + '_mismatch.png')


    def sector_level(self,fname,dpi): 
        self.dU_sector_level = pd.DataFrame(index = self.output.index,columns=self.input.BEA_sector.unique())
        self.M_sector_level  = pd.DataFrame(index = self.output.index,columns=self.input.BEA_sector.unique())
        for i in range(self.output.index.shape[0]):
            self.dU_sector_level.iloc[i,:] = (np.array(self.output.iloc[i,:self.param['Nsector']])-np.array(self.input.u[self.input.date==self.output.index[i]]))*100
            self.M_sector_level.iloc[i,:]  = Mindex_sectoral(np.array(self.input.u[self.input.date==self.output.index[i]]),np.array(self.output.iloc[i,:self.param['Nsector']]),np.array(self.input.v[self.input.date==self.output.index[i]]),self.param['φ'],self.param['η'],self.param['mfunc'])

        self.dU_sectoral_plot, ax = plt.subplots(1,1,dpi = dpi)
        for i, name in enumerate(self.dU_sector_level.columns):
            ax.plot(self.dU_sector_level.index, self.dU_sector_level.iloc[:,i], label=name)
            ax.legend(fontsize='xx-small',loc='lower left',ncol=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Change in Unemployment')
        ax.set_title('Sectoral Changes in Unemployment')
        plt.savefig(self.param['outpath'] + fname + '_sectoral_unemployment_change.png')

        self.M_sectoral_plot, ax = plt.subplots(1,1,dpi = dpi)
        for i, name in enumerate(self.dU_sector_level.columns):
            ax.plot(self.M_sector_level.index, self.M_sector_level.iloc[:,i], label=name)
            ax.legend(fontsize='xx-small',loc='lower left',ncol=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Mismatch Index')
        ax.set_title('Sectoral Mismatch Index')
        plt.savefig(self.param['outpath'] + fname + '_sectoral_mismatch_index.png')

        #scatter plots of lambda, alpha,lambda*alpha, e with v, phi
        self.sectoral_scatter(self.input.v,'v',fname+'_v',dpi)
        self.sectoral_scatter(np.tile(self.param['φ'],self.input.date.unique().shape[0]),'φ',fname+'_phi',dpi)

    def sectoral_scatter(self,x,xlab,fname,dpi):
        self.scatter, ax = plt.subplots(2,2,dpi=dpi)
        plt.rcParams['font.size'] = '6'
        ax[0,0].plot(x,np.tile(self.param['λ'],self.input.date.unique().shape[0]),'o')
        m,b = np.polyfit(x, np.tile(self.param['λ'],self.input.date.unique().shape[0]), 1)
        ax[0,0].plot(x,m*x+b,":r")
        ax[0,0].set_ylabel('λ')
        ax[0,0].set_xlabel(xlab)
        ax[0,1].plot(x,np.tile(self.param['α'],self.input.date.unique().shape[0]),'o')
        m,b = np.polyfit(x, np.tile(self.param['α'],self.input.date.unique().shape[0]), 1)
        ax[0,1].plot(x,m*x+b,":r")
        ax[0,1].set_ylabel('α')
        ax[0,1].set_xlabel(xlab)
        ax[1,0].plot(x,np.tile(self.param['λ']*self.param['α'],self.input.date.unique().shape[0]),'o')
        m,b = np.polyfit(x, np.tile(self.param['λ']*self.param['α'],self.input.date.unique().shape[0]), 1)
        ax[1,0].plot(x,m*x+b,":r")
        ax[1,0].set_ylabel('λα')
        ax[1,0].set_xlabel(xlab)
        ax[1,1].plot(x,self.input.e,'o')
        m,b = np.polyfit(x, self.input.e, 1)
        ax[1,1].plot(x,m*x+b,":r")
        ax[1,1].set_ylabel('e')
        ax[1,1].set_xlabel(xlab)
        plt.savefig(self.param['outpath'] + fname + '_sectoral_scatter.png')
    
    def social_welfare(self,param_sw,tol=1e-8,maxiter=1e5,ntrue=5,guessrange=100):
        self.param_sw = param_sw
        self.param_sw['θ'] = self.param['θ']
        self.param_sw['α'] = self.param['α']
        self.param_sw['A'] = self.param['A']
        
        # For now check only most recent observation in the series 
        v = np.array(self.input.v[self.input.date==self.input.date.unique()[-1]])
        u = np.array(self.input.u[self.input.date==self.input.date.unique()[-1]])
        e = np.array(self.input.e[self.input.date==self.input.date.unique()[-1]])
        ustar = np.array(self.output.iloc[-1,:v.shape[0]])

        # solution with old unemployment levels
        self.param_sw['L'] = e+self.param['mfunc'](u,v,self.param['φ'],self.param['η'])
        sol_u = root_robust(self.param_sw['objective'],self.param_sw,uguess_mean=0.2*np.ones(v.shape[0]**2+2*v.shape[0]-1),tol=tol,maxiter=maxiter,ntrue=ntrue,guessrange=guessrange)
        
        # solution with new unemployment levels 
        self.param_sw['L'] = e+self.param['mfunc'](ustar,v,self.param['φ'],self.param['η'])
        sol_ustar = root_robust(self.param_sw['objective'],self.param_sw,uguess_mean=np.ones(v.shape[0]**2+2*v.shape[0]-1),tol=tol,maxiter=maxiter,ntrue=ntrue,guessrange=guessrange)

        #self.Y     = 
        #self.Ystar = 
        return

# Filtering functions

def HP(y,λ):
    dim = y.shape[0]
    y = y.reshape(dim,1)
    H = HL(dim,λ)
    x = np.matmul(np.linalg.inv(H),y)
    c = y - x
    return x, c

def HL(dim,λ):
    #main diagonal
    d0 = np.ones((dim,))*(1+6*λ)
    d0[0], d0[1], d0[-2], d0[-1] = 1+λ, 1+5*λ, 1+5*λ, 1+λ
    #one above and one below main diagonal
    d1 = np.ones((dim-1,))*(-4*λ)
    d1[0], d1[-1] = -2*λ, -2*λ
    #two above and two below main diagonal
    d2 = np.ones((dim-2,))*λ
    H = np.diag(d2,-2) + np.diag(d1,-1) + np.diag(d0,0) + np.diag(d1,1) + np.diag(d2,2) 
    return H