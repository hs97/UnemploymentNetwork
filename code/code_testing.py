import numpy as np
import matplotlib.pyplot as plt
from functions.optimal_unemployment import m_cd, mu_cd, Lstar, Lones, objective, ustar, Mindex

#simple two industry testing environment 

runplot = False
run700  = False

#baseline parameters
eraw = np.array([1200,2400])
uraw = np.array([68,108])
vraw = np.array([62,43])
φ = np.array([1,0.5])
η = 0.5
λ = np.ones(2)
α = np.ones(2)

#adjusting so that everything sums to 1
e2 = eraw/np.sum(eraw+uraw)
u2 = uraw/np.sum(eraw+uraw)
v2 = vraw/np.sum(eraw+uraw)

# baseline testing
h = m_cd(u=u2,v=v2,φ=φ,η=η)
L = Lstar(u=u2,v=v2,e=e2,φ=φ,η=η,mfunc=m_cd)
ustar_baseline = ustar(objective,v2,e2,φ,η,λ,α,m_cd,mu_cd,Lones,uguess=u2,method='hybr',maxiter=1e4)
Mbaseline = Mindex(u2,ustar_baseline.x,v2,φ,η,m_cd)


# with a network structure
θ    = np.array([0.5,0.5])
A    = np.array([[0.4,0.1],[0.2,0.6]])
Linv = np.linalg.inv(np.identity(2)-A)
λ    = θ.reshape((1,2))@Linv 
λ    = λ.reshape((2,))
α    = np.ones(2)-np.sum(A,1)

h = m_cd(u=u2,v=v2,φ=φ,η=η)
L = Lstar(u=u2,v=v2,e=e2,φ=φ,η=η,mfunc=m_cd)
ustar_network = ustar(objective,v2,e2,φ,η,λ,α,m_cd,mu_cd,Lstar,uguess=u2,method='hybr',maxiter=1e4)
Mnetwork = Mindex(u2,ustar_network.x,v2,φ,η,m_cd)

#plotting the matching function for different values of v given some unemployment and some parameters, demonstrates that Cobb-Douglas can lead to more hires than vacancies or unemployed workers
if runplot:
    nrange = 10000
    minrange = 0.001
    maxrange = 5
    vrange = np.linspace(minrange,maxrange,nrange)
    hrange = m_cd(u=np.ones(nrange),v=vrange,φ=1,η=0.5)
    plt.plot(vrange,hrange)


# extending to 17 sector example, to check run time 
nsector = 17
eraw = np.random.uniform(1000,10000,(nsector,))
uraw = np.random.uniform(100,1000,(nsector,))
vraw = np.random.uniform(100,1000,(nsector,))
φ = np.random.uniform(0.1,1,(nsector,))
η = 0.5
λ = np.ones(nsector)
α = np.ones(nsector)

#adjusting so that everything sums to 1
e17 = eraw/np.sum(eraw+uraw)
u17 = uraw/np.sum(eraw+uraw)
v17 = vraw/np.sum(eraw+uraw)

# large testing
h17 = m_cd(u=u17,v=v17,φ=φ,η=η)
L17 = Lstar(u=u17,v=v17,e=e17,φ=φ,η=η,mfunc=m_cd)
ustar_17 = ustar(objective,v17,e17,φ,η,λ,α,m_cd,mu_cd,Lones,uguess=u17,method='hybr',maxiter=1e4)
M17 = Mindex(u17,ustar_17.x,v17,φ,η,m_cd)


# testing robustness to initial guess, probably a good idea to implement in full code
nguess = 1000
guessmat = np.random.uniform(0.0001,0.1,(nguess,nsector))
guess_success = np.zeros(nguess)
guess_sol     = np.zeros_like(guessmat)

for i in range(nguess):
    ustar_guess = ustar(objective,v17,e17,φ,η,λ,α,m_cd,mu_cd,Lones,uguess=guessmat[i,:],method='hybr',maxiter=1e4)
    guess_success[i] = ustar_guess.success
    guess_sol[i,:] = ustar_guess.x
    print('attempt ' + str(i) + ':')
    print(ustar_guess.success)

guess_sol_suc = np.zeros((int(np.sum(guess_success)),nsector))

ind = 0
for i in range(nguess):
    if guess_success[i]==1:
        guess_sol_suc[ind,:] = guess_sol[i,:]
        ind = ind + 1

# extending to 100 sector example, to check run time 
if run700:
    nsector = 700
    eraw = np.random.uniform(1000,10000,(nsector,))
    uraw = np.random.uniform(100,1000,(nsector,))
    vraw = np.random.uniform(100,1000,(nsector,))
    φ = np.random.uniform(0.1,1,(nsector,))
    η = 0.5
    λ = np.ones(nsector)
    α = np.ones(nsector)

    #adjusting so that everything sums to 1
    e100 = eraw/np.sum(eraw+uraw)
    u100 = uraw/np.sum(eraw+uraw)
    v100 = vraw/np.sum(eraw+uraw)

    # large testing
    h100 = m_cd(u=u100,v=v100,φ=φ,η=η)
    L100 = Lstar(u=u100,v=v100,e=e100,φ=φ,η=η,mfunc=m_cd)
    ustar_100 = ustar(objective,v100,e100,φ,η,λ,α,m_cd,mu_cd,Lones,uguess=u100,method='hybr',maxiter=1e4)
    M100 = Mindex(u100,ustar_100.x,v100,φ,η,m_cd)


    # testing robustness to initial guess, probably a good idea to implement in full code
    nguess = 1000
    guessmat = np.random.uniform(0.0001,0.1,(nguess,nsector))
    guess_success = np.zeros(nguess)
    guess_sol     = np.zeros_like(guessmat)

    for i in range(nguess):
        ustar_guess = ustar(objective,v100,e100,φ,η,λ,α,m_cd,mu_cd,Lones,uguess=guessmat[i,:],method='hybr',maxiter=1e4)
        guess_success[i] = ustar_guess.success
        guess_sol[i,:] = ustar_guess.x
        print('attempt ' + str(i) + ':')
        print(ustar_guess.success)

    guess_sol_suc = np.zeros((int(np.sum(guess_success)),nsector))

    ind = 0
    for i in range(nguess):
        if guess_success[i]==1:
            guess_sol_suc[ind,:] = guess_sol[i,:]
            ind = ind + 1

print('done')

