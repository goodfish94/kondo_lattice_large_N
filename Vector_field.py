# solution of vector fields

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class phi_fields:
    def __init__(self,para):
        self.para = para
        self.c = para['c']
        self.g = para['g']
        self.Qphi = para['Qphi']
        self.beta = para['beta']

        self.ir_cut = para['ir_cut'] # render singularity
        self.lambda_ = para['lambda'] # uv cut


        self.gc = (self.Qphi * 4*np.pi/self.lambda_/self.c)**2

        self.Np = 1000
        self.p = np.linspace(self.ir_cut, self.lambda_,self.Np)
        self.dp = self.p[1] - self.p[0]

    def get_self_consistent_eq (self, eta):
        x = np.sqrt(self.p**2 * self.c**2/self.g + eta**2 * self.c**2 )
        re = np.sum( self.p * self.dp /4/np.pi * self.c**2 /np.tanh(x/2.0)/(2.0*x) )
        re = re - self.Qphi
        return re

    def get_sol(self,eta0 = 0.5):
        fun = lambda x: self.get_self_consistent_eq(x)
        sol = optimize.root( fun, eta0, method='broyden1', options={'fatol':1e-3,'maxiter':1000})
        if( sol.success == False ):
            return 0.0
        else:
            return sol.x**2



if __name__ == '__main__':
    para = {'c':1.0, 'Qphi':1.0,'beta':1000, 'g':1.0, 'ir_cut':1e-4, 'lambda':0.1*2*np.pi }
    phi_sol = phi_fields(para)
    print(phi_sol.gc)

    for g in np.linspace(0.1,10,3):
        print("g = ",g)
        print("eta = ", phi_sol.get_sol())
