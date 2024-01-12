"""
Script that runs the following suite of verification
tests & checks
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py,os

from Transforms import Test_Cosine_Transform_NL, Test_Cosine_Transform_deal
from Transforms import Test_Sine_Transform_NL, Test_Sine_Transform_deal

from Matrix_Operators import cheb_radial
from Main import _Time_Step,_Newton, Nusselt, Kinetic_Energy

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
    Tau = 1.; Ra_s = 500.0; Pr = 1.; # Parameters set at the top of Main.py
    d   = 2.; Ra   = 7268.365;
    N_fm = 10;N_r = 20;

    μ_args = [Ra,Ra_s,Tau,Pr]

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    X = np.random.rand(3*N);
    X = 1e-03*(X/np.linalg.norm(X,2))

    Time_steps = [5e-3,2.5e-03,1.25e-03]#,6.125e-04]
    Slopes     = [];
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        _Time_Step(X,μ_args,N_fm,N_r,d, save_filename=filename,start_time=0, Total_time=20, dt=dt, symmetric =False, linear=True, Verbose=False);
        Slopes.append( slope(filename) )

    print('\n')
    print('dt     = ',Time_steps)
    print('lambda = ',Slopes)
    print('\n')

    return None;

def Nonlinear_Tstep__test():

    #~~~~~~~~~~~ Wide Gap l=2 ~~~~~~~~~~~
    Tau = 1.; Ra_s = 0.0; Pr = 10.;  
    d   = 2.; Ra   = 6780.; 

    N_fm = 32;
    N_r  = 16;
    dt   = 0.075;

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    X = np.random.rand(3*N);
    X = 1e-03*(X/np.linalg.norm(X,2))

    filename = 'Non_Linear_Test_wide_gap_dt'+str(dt)+'.h5'
    
    μ_args = [Ra,Ra_s,Tau,Pr]
    X_new  = _Time_Step(X,μ_args, N_fm,N_r,d, save_filename=filename,start_time=0., Total_time=2*(10**3), dt=dt, symmetric=False,linear=False,Verbose=False);

    T_hat  = X_new[N:2*N];
    Nu_avg = Nusselt(T_hat, d,     R,D,N_fm,nr);
    KE_avg = Kinetic_Energy(X_new, R,D,N_fm,nr);

    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',Nu_avg)
    print('<KE> = ',KE_avg)
    print('\n')

    #~~~~~~~~~~~ Narrow Gap l=10 ~~~~~~~~~~~
    Tau = 1.;    Ra_s = 0.0; Pr = 1.;  
    d   = 0.353; Ra   = 2360.0; 

    N_fm = 48;
    N_r  = 24;
    dt   = 0.075;

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    X = np.random.rand(3*N);
    X = 1e-03*(X/np.linalg.norm(X,2))

    filename = 'Non_Linear_Test_thin_gap_dt'+str(dt)+'.h5'
    
    μ_args = [Ra,Ra_s,Tau,Pr]
    X_new  = _Time_Step(X,μ_args, N_fm,N_r,d, save_filename=filename,start_time=0., Total_time=2*(10**3), dt=dt, symmetric=False,linear=False,Verbose=False);

    T_hat  = X_new[N:2*N];
    Nu_avg = Nusselt(T_hat, d,     R,D,N_fm,nr);
    KE_avg = Kinetic_Energy(X_new, R,D,N_fm,nr);

    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',Nu_avg)
    print('<KE> = ',KE_avg)
    print('\n')

    return None;

def Nonlinear_Newton_test():

    #~~~~~~~~~~~ Wide Gap l=2 ~~~~~~~~~~~
    Tau = 1.; Ra_s = 0.0; Pr = 10.;  
    d   = 2.; Ra   = 6780.; 

    N_fm = 32;
    N_r  = 16;
    dt   = 0.075;

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    filename = 'Non_Linear_Test_wide_gap_dt'+str(dt)+'.h5'; frame =-1;
    if filename.endswith('.h5'):
        
        f = h5py.File(filename, 'r+')

        # Problem Params
        X      = f['Checkpoints/X_DATA'][frame];
        Ra     = f['Parameters']["Ra"][()];
        N_fm   = f['Parameters']["N_fm"][()]
        N_r    = f['Parameters']["N_r"][()]
        d 	   = f['Parameters']["d"][()]
        #start_time = f['Parameters']["start_time"][()];
        try:
            start_time  = f['Scalar_Data/Time'][()][frame]
        except:
            pass;
        f.close();

        print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(start_time,Ra,d,N_fm,N_r))
    
    μ_args = [Ra,Ra_s,Tau,Pr];
    X_new,ke,nuS,nuT,BOOL = _Newton(X,μ_args, N_fm,N_r,d, Krylov_Space_Size = 150, symmetric = True)
    
    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',nuT)
    print('<KE> = ',ke)
    print('\n')

    #~~~~~~~~~~~ Narrow Gap l=10 ~~~~~~~~~~~
    Tau = 1.;    Ra_s = 0.0; Pr = 1.;  
    d   = 0.353; Ra   = 2360.0; 

    N_fm = 48;
    N_r  = 24;
    dt   = 0.075;

    D,R  = cheb_radial(N_r,d); 
    nr   = len(R[1:-1]);
    N 	 = nr*N_fm;

    filename = 'Non_Linear_Test_thin_gap_dt'+str(dt)+'.h5'; frame =-1;
    if filename.endswith('.h5'):
        
        f = h5py.File(filename, 'r+')

        # Problem Params
        X      = f['Checkpoints/X_DATA'][frame];
        Ra     = f['Parameters']["Ra"][()];
        N_fm   = f['Parameters']["N_fm"][()]
        N_r    = f['Parameters']["N_r"][()]
        d 	   = f['Parameters']["d"][()]
        #start_time = f['Parameters']["start_time"][()];
        try:
            start_time  = f['Scalar_Data/Time'][()][frame]
        except:
            pass;
        f.close();

        print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(start_time,Ra,d,N_fm,N_r))
    
    μ_args = [Ra,Ra_s,Tau,Pr];
    X_new,ke,nuS,nuT,BOOL = _Newton(X,μ_args, N_fm,N_r,d, Krylov_Space_Size = 150, symmetric = True)
    
    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',nuT)
    print('<KE> = ',ke)
    print('\n')

    return None;

if __name__ == "__main__":

    print('Creating a test directory .... \n')
    os.rmdir('./Tests')
    os.mkdir('./Tests')
    os.chdir('./Tests')

    print('Running Transforms Tests ..... \n')
    N = 2**8;
    Test_Cosine_Transform_NL(N)
    Test_Sine_Transform_NL(N)

    for k in range(3):
        Test_Cosine_Transform_deal(k,N);
        Test_Sine_Transform_deal(k+1,N);

    print('Running Linear Tests ..... \n')
    Linear_test();

    print('Running Non-Linear Tests ..... \n')
    Nonlinear_Tstep__test()
    Nonlinear_Newton_test()