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


def mismatch_estimation(df,objective,φ,η,λ,α,mfunc,mufunc,Lfunc,tol=1e-6,maxiter=1e5,ntrue=100,guessrange=0.1,HP_lam=129000):
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
        print('Starting date: ' + str(df.date.unique()[i]))
        print('Date successful: ' + str(success))
    
    output['mismatch index'] = M_t
    M_x, M_c = HP(M_t,HP_lam)
    output['mismatch trend'] = M_x
    output['mismatch cycle'] = M_c

    return output

def ustar_objective(uopt,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc):
    
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

def root_robust(objective,v,e,φ,η,λ,α,mfunc,mufunc,Lfunc,uguess_mean=np.array([]),tol=1e-6,maxiter=1e4,ntrue=100,guessrange=0.1):
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
            #print('Num converged: ' + str(count_true))
    out_mat = out_mat.reshape((ntrue,v.shape[0]))
    out_gap = out_mat - out_mat[0,:]
    if np.max(np.abs(out_gap))>10*tol:
        success = False
    else:
        success = True
    return  np.mean(out_mat,axis=0), success

    self.v_scatter, ax = plt.subplots(2,2,dpi=dpi)
        plt.rcParams['font.size'] = '6'
        ax[0,0].plot(self.input.v,np.tile(self.param['λ'],self.input.date.unique().shape[0]),'o')
        z = np.polyfit(self.input.v, np.tile(self.param['λ'],self.input.date.unique().shape[0]), 1)
        p = np.poly1d(z)
        ax[0,0].plot(self.input.v,p(self.input.v),"r--")
        ax[0,0].set_ylabel('λ')
        ax[0,0].set_xlabel('v')
        ax[0,1].plot(self.input.v,np.tile(self.param['α'],self.input.date.unique().shape[0]),'o')
        z = np.polyfit(self.input.v, np.tile(self.param['α'],self.input.date.unique().shape[0]), 1)
        p = np.poly1d(z)
        ax[0,1].plot(self.input.v,p(self.input.v),"r--")
        ax[0,1].set_ylabel('α')
        ax[0,1].set_xlabel('v')
        ax[1,0].plot(self.input.v,np.tile(self.param['λ']*self.param['α'],self.input.date.unique().shape[0]),'o')
        z = np.polyfit(self.input.v, np.tile(self.param['λ']*self.param['α'],self.input.date.unique().shape[0]), 1)
        p = np.poly1d(z)
        ax[1,0].plot(self.input.v,p(self.input.v),"r--")
        ax[1,0].set_ylabel('λα')
        ax[1,0].set_xlabel('v')
        ax[1,1].plot(self.input.v,self.input.e,'o')
        z = np.polyfit(self.input.v, self.input.e, 1)
        p = np.poly1d(z)
        ax[1,1].plot(self.input.v,p(self.input.v),"r--")
        ax[1,1].set_ylabel('e')
        ax[1,1].set_xlabel('v')
        plt.savefig(self.param['outpath'] + fname + '_v_sectoral_scatter.png')

        self.φ_scatter, ax = plt.subplots(2,2,dpi=dpi)
        plt.rcParams['font.size'] = '6'
        ax[0,0].plot(self.param['φ'],self.param['λ'],'o')
        ax[0,0].set_ylabel('λ')
        ax[0,0].set_xlabel('φ')
        ax[0,1].plot(self.param['φ'],self.param['α'],'o')
        ax[0,1].set_ylabel('α')
        ax[0,1].set_xlabel('φ')
        ax[1,0].plot(self.param['φ'],self.param['λ']*self.param['α'],'o')
        ax[1,0].set_ylabel('λα')
        ax[1,0].set_xlabel('φ')
        ax[1,1].plot(self.param['φ'],self.param['e'],'o')
        ax[1,1].set_ylabel('e')
        ax[1,1].set_xlabel('φ')
        plt.savefig(self.param['outpath'] + fname + '_phi_sectoral_scatter.png')