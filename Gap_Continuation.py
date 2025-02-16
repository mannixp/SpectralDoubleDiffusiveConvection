from Main import _Continuation, _plot_bif
import h5py
import numpy as np
import os


def Gap_Vary(Ra_s_new, open_filename, frame):

    f = h5py.File(open_filename, 'r+')
    X = f['Checkpoints/X_DATA'][frame]

    try:
        Ra = f['Checkpoints/Ra_DATA'][frame]
        #Ra = f['Bifurcation/Ra_DATA'][frame];
    except:
        Ra = f['Parameters']["Ra"][()]

    Ra_s = f['Parameters']["Ra_s"][()]
    Tau = f['Parameters']["Tau"][()]
    Pr = f['Parameters']["Pr"][()]
    d = f['Parameters']["d"][()]
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    f.close()

    # ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
    from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
    N_r_n = 32
    X = INTERP_RADIAL(N_r_n, N_r, X, d)
    N_fm_n = 256
    X = INTERP_THETAS(N_fm_n, N_fm, X)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n Loading Ra = %e, Ra_s=%e, Pr=%2.5f, Tau=%2.5f, d=%2.5f and resolution N_fm, N_r = %d,%d \n"%(Ra,Ra_s,Pr,Tau,d,N_fm,N_r))    
    
    sign = -1
    N_steps = 350
    Y = np.hstack((X, Ra))
    kwargs = {"Ra":Ra,"Ra_s":Ra_s_new,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm_n,"N_r":N_r_n, "symmetric":True}
    
    # Generate a new path
    filename, extension = os.path.splitext(open_filename)
    counter = int(filename.split('_')[1])
    save_filename = filename.split('_')[0] + "Ras" + str(Ra_s_new) + "_" + str(counter) + extension

    _Continuation(save_filename, N_steps, sign, Y, **kwargs)
    _plot_bif(save_filename)

    return save_filename


def main():

    print('Creating a test directory .... \n')

    open_filename = "ConvectonL10PlusRas200_1.h5"
    frame = -1
    Ra_s = 175
    open_filename = Gap_Vary(Ra_s, open_filename, frame)

    # for Ra_s in range(200, 450, 50):
    #     print("\n Ra_s new = %e \n" % Ra_s)
    #     open_filename = Gap_Vary(Ra_s, open_filename, frame)
    
    return None


def main_Large():

    print('Creating a test directory .... \n')

    open_filename = "Continuationl11Test_2.h5"
    frame = 30
   
    for Ra_s in range(500, 100, -50):
        print("\n Ra_s new = %e \n" % Ra_s)
        Gap_Vary(Ra_s, open_filename, frame)

    return None


if __name__ == "__main__":

    # %%
    print('Setup')

    # %%
    main()

    # %%
    #main_Large()
# %%
