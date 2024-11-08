"""
Script that runs the following suite of verification
tests & checks
"""
import numpy as np
import h5py
import os
from Matrix_Operators import cheb_radial
from Main import _Time_Step, _Newton, Nusselt, Kinetic_Energy
from Transforms import test_Cosine_Transform_NL, test_Cosine_Transform_deal
from Transforms import test_Sine_Transform_NL, test_Sine_Transform_deal


def slope(filename):
    """Compute the slope to estimate the growth rate."""
    st_pt = 1000
    f = h5py.File(filename, 'r')
    Time = f['Scalar_Data/Time'][()][-st_pt:]
    KE = f['Scalar_Data/Norm'][()][-st_pt:]
    f.close()

    slope, _ = np.polyfit(Time, np.log(KE), 1)

    return slope


def test_Linear_even():

    symmetric = True

    N_fm = 10
    N_r = 20

    nr = N_r - 1
    N = nr*N_fm

    # Validation Case l=2, Pr = 1, Tau = 1, Ra_s = 500
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:
        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 7268.365, "Ra_s": 500, "Tau": 1, "Pr": 1, "d": 2,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=2, Pr=1, Tau = 1, Ra_s = 500')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.0018195)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

    # Validation Case l=2, Pr=1, Tau = 10, Ra_s = 500
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 6900, "Ra_s": 500, "Tau": 10., "Pr": 1, "d": 2,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=2, Pr=1, Tau = 10, Ra_s = 500')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.14937553)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~


    # Validation Case l=2, Pr=1, Tau = 1/2, Ra_s = 500
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 7800, "Ra_s": 500, "Tau": 1/2, "Pr": 1, "d": 2,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=2, Pr=1, Tau = 1/2, Ra_s = 500')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.06596539)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~


    # Validation Case l=2, Pr=1, Tau = 1, Ra_s = 250
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 7100, "Ra_s": 250, "Tau": 1, "Pr": 1, "d": 2,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=2, Pr=1, Tau = 1, Ra_s = 250')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.15005375)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

    return None


def test_Nonlinear_Tstep_even():

    #~~~~~~~~~~~ Wide Gap l=2 ~~~~~~~~~~~
    d = 2
    N_fm = 32
    N_r = 16
    dt = 0.075

    D, R = cheb_radial(N_r, d)
    nr = len(R[1:-1])
    N = nr*N_fm

    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    filename = 'Non_Linear_Test_wide_gap_dt'+str(dt)+'.h5'
    kwargs = {"Ra": 6780, "Ra_s": 0, "Tau": 1, "Pr": 10, "d": d, "N_fm": N_fm, "N_r": N_r}
    X_new = _Time_Step(X, **kwargs, symmetric=True, save_filename=filename, start_time=0, Total_time=2*(10**3), dt=dt, linear=False, Verbose=False)

    T_hat = X_new[N:2*N]
    Nu_avg = Nusselt(T_hat, d,     R, D, N_fm, nr, check=False)
    KE_avg = Kinetic_Energy(X_new, R, D, N_fm, nr, symmetric=True)

    print('\n')
    print('Ra,Pr,d =%d,%d,%d' % (kwargs["Ra"], kwargs["Pr"], d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f' % (N_r, N_fm, dt))
    print('<Nu> = ', Nu_avg)
    print('<KE> = ', KE_avg)
    print('\n')

    #~~~~~~~~~~~ Narrow Gap l=10 ~~~~~~~~~~~
    d = 0.353
    N_fm = 48
    N_r = 24
    dt = 0.075

    D, R = cheb_radial(N_r, d)
    nr = len(R[1:-1])
    N = nr*N_fm

    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    filename = 'Non_Linear_Test_thin_gap_dt'+str(dt)+'.h5'
    kwargs = {"Ra": 2360, "Ra_s": 0, "Tau": 1, "Pr": 1, "d": d, "N_fm": N_fm, "N_r": N_r}
    X_new = _Time_Step(X, **kwargs, symmetric=True, save_filename=filename, start_time=0, Total_time=2*(10**3), dt=dt, linear=False, Verbose=False)

    T_hat = X_new[N:2*N]
    Nu_avg = Nusselt(T_hat, d, R, D, N_fm, nr, check=False)
    KE_avg = Kinetic_Energy(X_new, R, D, N_fm, nr, symmetric=True)

    print('\n')
    print('Ra,Pr,d =%d,%d,%d' % (kwargs["Ra"], kwargs["Pr"], d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f' % (N_r, N_fm, dt))
    print('<Nu> = ', Nu_avg)
    print('<KE> = ', KE_avg)
    print('\n')

    return None


def test_Nonlinear_Newton_even():

    dt   = 0.075;

    #~~~~~~~~~~~ Wide Gap l=2 ~~~~~~~~~~~
    filename = 'Non_Linear_Test_wide_gap_dt'+str(dt)+'.h5'; frame =-1;
    if filename.endswith('.h5'):
        
        f = h5py.File(filename, 'r+')

        # Problem Params
        X      = f['Checkpoints/X_DATA'][frame];

        Ra     = f['Parameters']["Ra"][()];
        Ra_s   = f['Parameters']["Ra_s"][()];
        Tau    = f['Parameters']["Tau"][()];
        Pr     = f['Parameters']["Pr"][()];
        d 	   = f['Parameters']["d"][()]

        N_fm   = f['Parameters']["N_fm"][()]
        N_r    = f['Parameters']["N_r"][()]

        st_time= f['Scalar_Data/Time'][()][frame]
        f.close();

        print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))    
    
    kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
    X_new,norm,ke,nuS,nuT,BOOL = _Newton(X,**kwargs,symmetric = True,tol_newton = 1e-8)

    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',nuT)
    print('<KE> = ',ke)
    print('\n')

    #~~~~~~~~~~~ Narrow Gap l=10 ~~~~~~~~~~
    filename = 'Non_Linear_Test_thin_gap_dt'+str(dt)+'.h5'; frame =-1;
    if filename.endswith('.h5'):
        
        f = h5py.File(filename, 'r+')

        # Problem Params
        X      = f['Checkpoints/X_DATA'][frame];

        Ra     = f['Parameters']["Ra"][()];
        Ra_s   = f['Parameters']["Ra_s"][()];
        Tau    = f['Parameters']["Tau"][()];
        Pr     = f['Parameters']["Pr"][()];
        d 	   = f['Parameters']["d"][()]

        N_fm   = f['Parameters']["N_fm"][()]
        N_r    = f['Parameters']["N_r"][()]

        st_time= f['Scalar_Data/Time'][()][frame]
        f.close();

        print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))
    
    kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
    X_new,norm,ke,nuS,nuT,BOOL = _Newton(X,**kwargs,symmetric = True,tol_newton = 1e-8)
    
    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',nuT)
    print('<KE> = ',ke)
    print('\n')

    return None;


def test_Linear_odd():

    symmetric = False
    N_fm = 16
    N_r = 20

    nr = N_r - 1
    N = nr*N_fm

    # Validation Case l=3, Pr = 1, Tau = 1, Ra_s = 500
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:
        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 5540, "Ra_s": 500, "Tau": 1, "Pr": 1, "d": 1.5,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=3, Pr=1, Tau = 1, Ra_s = 500')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.0152189)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

    # Validation Case l=3, Pr=1, Tau = 15, Ra_s = 500
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 5075, "Ra_s": 500, "Tau": 15, "Pr": 1, "d": 1.5,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=3, Pr=1, Tau = 15, Ra_s = 500')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.01964496)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~


    # Validation Case l=3, Pr=1, Tau = 1, Ra_s = 250
    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    Time_steps = [5e-3, 2.5e-03, 1.25e-03]#, 6.125e-04]
    Slopes = []
    for dt in Time_steps:

        filename = 'Linear_Test_dt'+str(dt)+'.h5'
        kwargs = {"Ra": 5285, "Ra_s": 250, "Tau": 1, "Pr": 1, "d": 1.5,
                  "N_fm": N_fm, "N_r": N_r}
        _Time_Step(X, **kwargs, save_filename=filename, start_time=0,
                   Total_time=40, dt=dt, symmetric=symmetric, linear=True, Verbose=False)
        Slopes.append(slope(filename))

    print('\n')
    print('Validation Case l=3, Pr=1, Tau = 1, Ra_s = 250')
    print('dt     = ', Time_steps)
    print('lambda = ', Slopes)
    print('Target = ', 0.0183044)
    print('\n')
    #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~

    return None


def test_Nonlinear_Tstep_odd():

    #~~~~~~~~~~~ Narrow Gap l=11 ~~~~~~~~~~~
    d = 0.31325
    N_fm = 48
    N_r = 24
    dt = 0.075

    D, R = cheb_radial(N_r, d)
    nr = len(R[1:-1])
    N = nr*N_fm

    X = np.random.rand(3*N)
    X = 1e-03*(X/np.linalg.norm(X, 2))

    filename = 'Non_Linear_Test_thin_gap_odd_dt'+str(dt)+'.h5'
    kwargs = {"Ra": 2280, "Ra_s": 0, "Tau": 1, "Pr": 1, "d": d, "N_fm": N_fm, "N_r": N_r}
    X_new = _Time_Step(X, **kwargs, symmetric=False, save_filename=filename, start_time=0, Total_time=3*(10**3), dt=dt, linear=False, Verbose=False)

    T_hat = X_new[N:2*N]
    Nu_avg = Nusselt(T_hat, d, R, D, N_fm, nr, check=False)
    KE_avg = Kinetic_Energy(X_new, R, D, N_fm, nr, symmetric=True)

    print('\n')
    print('Ra,Pr,d =%d,%d,%d' % (kwargs["Ra"], kwargs["Pr"], d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f' % (N_r, N_fm, dt))
    print('<Nu> = ', Nu_avg)
    print('<KE> = ', KE_avg)
    print('\n')

    return None


def test_Nonlinear_Newton_odd():

    dt   = 0.075;

    #~~~~~~~~~~~ Narrow Gap l=11 ~~~~~~~~~~
    filename = 'Non_Linear_Test_thin_gap_odd_dt'+str(dt)+'.h5'; frame =-1;
    if filename.endswith('.h5'):
        
        f = h5py.File(filename, 'r+')

        # Problem Params
        X      = f['Checkpoints/X_DATA'][frame];

        Ra     = f['Parameters']["Ra"][()];
        Ra_s   = f['Parameters']["Ra_s"][()];
        Tau    = f['Parameters']["Tau"][()];
        Pr     = f['Parameters']["Pr"][()];
        d 	   = f['Parameters']["d"][()]

        N_fm   = f['Parameters']["N_fm"][()]
        N_r    = f['Parameters']["N_r"][()]

        st_time= f['Scalar_Data/Time'][()][frame]
        f.close();

        print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))
    
    kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
    X_new,norm,ke,nuS,nuT,BOOL = _Newton(X,**kwargs,symmetric=False,tol_newton = 1e-8)
    
    print('\n')
    print('Ra,Pr,d =%d,%d,%d'%(Ra,Pr,d))
    print('N_r,N_θ,∆t =%d,%d,%2.3f'%(N_r,N_fm,dt))
    print('<Nu> = ',nuT)
    print('<KE> = ',ke)
    print('\n')

    return None;


if __name__ == "__main__":

    print('Creating a test directory .... \n')
    import shutil
    shutil.rmtree('./Tests', ignore_errors=True)
    os.mkdir('./Tests')
    os.chdir('./Tests')

    print('Running Transforms Tests ..... \n')
    N = 2**8
    test_Cosine_Transform_NL(N)
    test_Sine_Transform_NL(N)

    for k in range(3):
        test_Cosine_Transform_deal(k,N);
        test_Sine_Transform_deal(k+1,N);

    print('******** Even parity tests *********')
    test_Linear_even()
    test_Nonlinear_Tstep_even()
    test_Nonlinear_Newton_even()

    print('******** Odd parity tests *********')
    test_Linear_odd()
    test_Nonlinear_Tstep_odd()
    test_Nonlinear_Newton_odd()