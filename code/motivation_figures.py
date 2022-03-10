import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from functions.labor_markets import labor_market, economy

#### baseline employement, unemployment, and output with variable labor market mobility ####
b = 0.8
lamvec     = np.linspace(0.5,1,1000) #labor market openess from less open to less open
Lvec       = np.zeros((2,lamvec.shape[0]))
uratevec   = np.zeros_like(Lvec)
urateagg   = np.zeros_like(lamvec)
yvec       = np.zeros_like(Lvec)
for i in range(lamvec.shape[0]):
    m1 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai=0)
    m2 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai=b)
    eq = economy(m1,m2)
    eq.solve_equilibrium()
    Lvec[0,i], Lvec[1,i] = eq.market1.Ld, eq.market2.Ld
    uratevec[0,i], uratevec[1,i] = eq.market1.urate, eq.market2.urate
    urateagg[i] = (2 -  eq.market1.Ld- eq.market2.Ld)/2
    yvec[0,i], yvec[1,i] = eq.market1.yi, eq.market2.yi


fig_baseline, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lvec[0,:],'-k')
plot1.plot(lamvec,Lvec[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.invert_xaxis()

plot2.plot(lamvec,uratevec[0,:],'-k')
plot2.plot(lamvec,uratevec[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yvec[0,:],'-k')
plot3.plot(lamvec,yvec[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/baseline.png')

fig_unemployment_baseline, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,uratevec[0,:],'-k')
plot1.set_ylabel('Employment')
plot1.invert_xaxis()

plot2.plot(lamvec,uratevec[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,urateagg[:],'-k')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/unemployment_baseline.png')

# responses of equilibrium outcomes to positive 5% shock to productivity in upstream sector at varying levels of labor market mobility
shock = 1.01
Lvec_s1       = np.zeros((2,lamvec.shape[0]))
uratevec_s1   = np.zeros_like(Lvec)
urateagg_s1   = np.zeros_like(lamvec)
yvec_s1       = np.zeros_like(Lvec)
for i in range(lamvec.shape[0]):
    m1 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai=0, Ai=shock)
    m2 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai= b)
    eq = economy(m1,m2)
    eq.solve_equilibrium()
    Lvec_s1[0,i], Lvec_s1[1,i] = eq.market1.Ld, eq.market2.Ld
    uratevec_s1[0,i], uratevec_s1[1,i] = eq.market1.urate, eq.market2.urate
    urateagg_s1 =  (eq.market1.urate + eq.market2.urate)/2
    yvec_s1[0,i], yvec_s1[1,i] = eq.market1.yi, eq.market2.yi

fig_s1, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lvec_s1[0,:],'-k')
plot1.plot(lamvec,Lvec_s1[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.invert_xaxis()

plot2.plot(lamvec,uratevec_s1[0,:],'-k')
plot2.plot(lamvec,uratevec_s1[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yvec_s1[0,:],'-k')
plot3.plot(lamvec,yvec_s1[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/baseline_A1.png')

#side of shock to other parts
Lshock1    = (Lvec_s1-Lvec)/Lvec
ushock1    = (uratevec_s1-uratevec)/uratevec
yshoc1     = (yvec_s1-yvec)/yvec
uaggshock1 = (urateagg_s1-urateagg)/urateagg

fig_shock1, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lshock1[0,:],'-k')
plot1.plot(lamvec,Lshock1[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.set_title('Response to 1 percent shock to A1')
plot1.invert_xaxis()

plot2.plot(lamvec,ushock1[0,:],'-k')
plot2.plot(lamvec,ushock1[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yshoc1[0,:],'-k')
plot3.plot(lamvec,yshoc1[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/shock_A1.png')

fig_unemployment_shock1, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,ushock1[0,:],'-k')
plot1.set_ylabel('Sector 1')
plot1.invert_xaxis()
plot1.set_title("Response of unemployment to shock in sector 1")

plot2.plot(lamvec,ushock1[1,:],'-k')
plot2.set_ylabel('Sector 2')
plot2.invert_xaxis()

plot3.plot(lamvec,uaggshock1[:],'-k')
plot3.set_ylabel('Aggregate')
plot3.set_xlabel('More production linkages ->')
plot3.invert_xaxis()
plt.savefig('output/unemployment_shock1.png')

# responses of equilibrium outcomes to positive 5% shock to productivity in downstream sector at varying levels of labor market mobility
shock = 1.01
Lvec_s2       = np.zeros((2,lamvec.shape[0]))
uratevec_s2   = np.zeros_like(Lvec)
urateagg_s2   = np.zeros_like(lamvec)
yvec_s2       = np.zeros_like(Lvec)
for i in range(lamvec.shape[0]):
    m1 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai=0)
    m2 = labor_market(lambdai=lamvec[i], lambdaj=lamvec[i], betai=b, Ai=shock)
    eq = economy(m1,m2)
    eq.solve_equilibrium()
    Lvec_s2[0,i], Lvec_s2[1,i] = eq.market1.Ld, eq.market2.Ld
    uratevec_s2[0,i], uratevec_s2[1,i] = eq.market1.urate, eq.market2.urate
    urateagg_s2 = (eq.market1.urate + eq.market2.urate)/2
    yvec_s2[0,i], yvec_s2[1,i] = eq.market1.yi, eq.market2.yi

fig_s1, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lvec_s2[0,:],'-k')
plot1.plot(lamvec,Lvec_s2[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.invert_xaxis()

plot2.plot(lamvec,uratevec_s2[0,:],'-k')
plot2.plot(lamvec,uratevec_s2[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yvec_s2[0,:],'-k')
plot3.plot(lamvec,yvec_s2[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/baseline_A2.png')

#side of shock to other parts
Lshock2 = (Lvec_s2-Lvec)/Lvec
ushock2 = (uratevec_s2-uratevec)/uratevec
yshoc2 = (yvec_s2-yvec)/yvec
uaggshock2 = (urateagg_s2-urateagg)/urateagg


fig_shock2, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lshock2[0,:],'-k')
plot1.plot(lamvec,Lshock2[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.set_title('Response to 1 percent shock to A2')
plot1.invert_xaxis()

plot2.plot(lamvec,ushock2[0,:],'-k')
plot2.plot(lamvec,ushock2[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yshoc2[0,:],'-k')
plot3.plot(lamvec,yshoc2[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/shock_A2.png')

fig_unemployment_shock2, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,ushock2[0,:],'-k')
plot1.set_ylabel('Sector 1')
plot1.invert_xaxis()
plot1.set_title("Response of unemployment to shock in sector 2")

plot2.plot(lamvec,ushock2[1,:],'-k')
plot2.set_ylabel('Sector 2')
plot2.invert_xaxis()

plot3.plot(lamvec,uaggshock2[:],'-k')
plot3.set_ylabel('Aggregate')
plot3.set_xlabel('More production linkages ->')
plot3.invert_xaxis()
plt.savefig('output/unemployment_shock2.png')

#what if we do the same thing but for a variation in beta given the same labor market tightness. 
bvec  = np.linspace(0,1,1000)
lam   = 0.5
shock = 1.01 
Lvec_b       = np.zeros((2,bvec.shape[0]))
uratevec_b   = np.zeros_like(Lvec)
urateagg_b   = np.zeros_like(lamvec)
yvec_b       = np.zeros_like(Lvec)
for i in range(lamvec.shape[0]):
    m1 = labor_market(lambdai=lam, lambdaj=lam, betai=0)
    m2 = labor_market(lambdai=lam, lambdaj=lam, betai=bvec[i])
    eq = economy(m1,m2)
    eq.solve_equilibrium()
    Lvec_b[0,i], Lvec_b[1,i] = eq.market1.Ld, eq.market2.Ld
    uratevec_b[0,i], uratevec_b[1,i] = eq.market1.urate, eq.market2.urate
    urateagg_b = (2 -  eq.market1.Ld- eq.market2.Ld)/2
    yvec_b[0,i], yvec_b[1,i] = eq.market1.yi, eq.market2.yi


Lvec_bs       = np.zeros((2,bvec.shape[0]))
uratevec_bs   = np.zeros_like(Lvec)
urateagg_bs   = np.zeros_like(lamvec)
yvec_bs       = np.zeros_like(Lvec)
for i in range(lamvec.shape[0]):
    m1 = labor_market(lambdai=lam, lambdaj=lam, betai=0,Ai=shock)
    m2 = labor_market(lambdai=lam, lambdaj=lam, betai=bvec[i])
    eq = economy(m1,m2)
    eq.solve_equilibrium()
    Lvec_bs[0,i], Lvec_bs[1,i] = eq.market1.Ld, eq.market2.Ld
    uratevec_bs[0,i], uratevec_bs[1,i] = eq.market1.urate, eq.market2.urate
    urateagg_bs[i] =  (eq.market1.urate + eq.market2.urate)/2
    yvec_bs[0,i], yvec_bs[1,i] = eq.market1.yi, eq.market2.yi

fig_s1, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lvec_b[0,:],'-k')
plot1.plot(lamvec,Lvec_b[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.invert_xaxis()

plot2.plot(lamvec,uratevec_b[0,:],'-k')
plot2.plot(lamvec,uratevec_b[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yvec_b[0,:],'-k')
plot3.plot(lamvec,yvec_b[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/baseline_b.png')

#side of shock to other parts
Lshockb = (Lvec_bs-Lvec_b)/Lvec_b
ushockb = (uratevec_bs-uratevec_b)/uratevec_b
yshocb = (yvec_bs-yvec_b)/yvec_b
uaggshockb = (urateagg_bs-urateagg_b)/urateagg_b


fig_shockb, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(lamvec,Lshockb[0,:],'-k')
plot1.plot(lamvec,Lshockb[1,:],'-r')
plot1.set_ylabel('Employment')
plot1.set_title('Response to 1 percent shock to A1')
plot1.invert_xaxis()

plot2.plot(lamvec,ushockb[0,:],'-k')
plot2.plot(lamvec,ushockb[1,:],'-r')
plot2.set_ylabel('Unemployment')
plot2.invert_xaxis()

plot3.plot(lamvec,yshocb[0,:],'-k')
plot3.plot(lamvec,yshocb[1,:],'-r')
plot3.set_ylabel('Output')
plot3.set_xlabel('More labor flexibility ->')
plot3.invert_xaxis()
plt.savefig('output/shock_b.png')

fig_unemployment_shockb, (plot1, plot2, plot3) = plt.subplots(3, 1)
plot1.plot(bvec,ushockb[0,:],'-k')
plot1.set_ylabel('Sector 1')
plot1.set_title("Response of unemployment to shock in sector 1")

plot2.plot(bvec,ushockb[1,:],'-k')
plot2.set_ylabel('Sector 2')

plot3.plot(bvec,uaggshockb[:],'-k')
plot3.set_ylabel('Aggregate')
plot3.set_xlabel('More production linkages ->')
plt.savefig('output/unemployment_shockb.png')

print('done')
