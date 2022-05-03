import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from functions.production_functions import cobb_douglas, CES

#Cobb-Douglas production function parameters
cd_params = {'A':1,'alpha':0.3,'a':np.array([0.4,0.3])}

F_cd = cobb_douglas(cd_params)

inputs = np.array([1,1,1])
y_cd = F_cd.output(inputs)

# CES production function parameters
ces_params = {'mu':0.5,'sigma':0.7,'xi':0.7,'a':np.array([0.5,0.5])}
F_ces = CES(ces_params)

inputs = np.array([1,1,1])
y_ces = F_ces.output(inputs)

Lrange = np.linspace(0.05,100,3000)
yrange = np.zeros_like(Lrange)
for i in range(Lrange.shape[0]):
    yrange[i] = F_ces.output(np.array([Lrange[i],1,1]))

plt.plot(Lrange,yrange)
plt.show()

print('done')

