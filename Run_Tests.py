"""
Script that runs the following suite of verification
tests & checks
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

from Matrix_Operators import cheb_radial
from Main import _Time_Step


def slope(filename):
    
    st_pt = 1000
    f = h5py.File(filename, 'r');
    Time  = f['Scalar_Data/Time'][()][-st_pt:]
    KE    = f['Scalar_Data/Norm'][()][-st_pt:]
    f.close()

    slope,_ = np.polyfit(Time,np.log(KE),1)

    return slope;

def Linear_test():

    # Validation Case l=2
    #Tau = 1.; Ra_s = 500.0; Pr = 1.; # Parameters set at the top of Main.py
    d  = 2.0; Ra  = 7268.365;
    N_fm = 10;N_r = 20;

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    X = np.random.rand(3*N);
    X = 1e-03*(X/np.linalg.norm(X,2))

    Time_steps = [5e-3,2.5e-03,1.25e-03]
    Slopes     = [];
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        _Time_Step(X,Ra,N_fm,N_r,d, save_filename=filename,start_time=0, Total_time=20, dt=dt, symmetric =True);
        Slopes.append( slope(filename) )

    print('\n')
    print('dt     = ',Time_steps)
    print('lambda = ',Slopes)
    print('\n')

    return None;


if __name__ == "__main__":

    Linear_test();