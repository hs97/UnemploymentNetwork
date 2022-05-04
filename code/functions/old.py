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