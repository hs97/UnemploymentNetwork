# Cobb-Douglas production function
import numpy as np
import scipy.optimize as opt

class cobb_douglas:
    def __init__(self,params):
        self.alpha = params['alpha']
        self.a = params['a'] 
        self.A = params['A']
        self.params = params
        if sum(self.a) + self.alpha < 1: 
            self.returns_to_scale = 'decreasing'
        elif sum(self.a) + self.alpha == 1:
            self.returns_to_scale = 'constant'
        else:
            self.returns_to_scale = 'increasing'

    def output(self,input):
        L = input[0]
        x = input[1:]
        
        y = self.A * np.power(L,self.alpha) * M_CD(x,self.a)

        self.L, self.x, self.y = L, x, y

        return y



class CES:
    def __init__(self,params):
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.xi = params['xi']
        self.a = params['a']
        self.params = params

    def output(self,input):
        L = input[0]
        x = input[1:]

        power1 = 1/self.sigma
        power2 = (self.sigma-1)/self.sigma

        y = ((1-self.mu)**power1 * L**power2 + self.mu**power1 * M_CES(x,self.a,self.xi)**power2)**(1/power2)
        
        self.L, self.x, self.y = L, x, y

        return y

def M_CD(x,a):
    return np.prod(np.power(x,a))



def M_CES(x,a,xi):
    return np.sum((a**(1/xi) * x**((xi-1)/xi)))**(xi/(xi-1))

