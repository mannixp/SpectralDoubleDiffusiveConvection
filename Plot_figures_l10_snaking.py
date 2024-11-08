import numpy as np
import glob, h5py
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Plot_Tools import Spectral_To_Gridpoints
from Main import result, Kinetic_Energy
from Matrix_Operators import cheb_radial
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


d = 0.3521  # Fixed gap-width
RES = 25  # number of contour levels to use
markersize = 10


def Plot_full_bif(folder, ax, line='k-'):
    """
    Plot out the bifurcation diagram and locate all the states
    corresponding to the fold points
    """ 
    def add_to_fig(obj):
        index = np.where(obj.Ra_dot[:-1]*obj.Ra_dot[1:] < 0)
        _, idx = np.unique(np.round(obj.Ra[index], 3), return_index=True)
        idx = np.sort(idx)
        Ra_f = obj.Ra[index][idx]
        KE_f = obj.KE[index][idx]

        #ax.semilogy(obj.Ra, obj.KE, line)
        #ax.semilogy(Ra_f, KE_f, 'ro', markersize=markersize)

        ax.plot(obj.Ra, obj.KE, line)
        ax.plot(Ra_f, KE_f, 'ro', markersize=markersize)

        # Return Saddles
        if len(obj.Y_FOLD) != 0:
            return obj.Y_FOLD[idx], obj.Ra[index][idx]
        else:
            return [], []
        
    X_fold = []
    Ra_fold = []
    N_fm_fold = []
    N_r_fold = []
    for filename in glob.glob(folder + '/*.h5'):
      
        obj = result()
        with h5py.File(filename, 'r') as f:
            ff = f["Bifurcation"]
            for key in ff.keys():
                setattr(obj, key, ff[key][()])
            N_fm = f['Parameters']["N_fm"][()]
            N_r = f['Parameters']["N_r"][()]

            X_f, Ra_f = add_to_fig(obj)
            for X_i, Ra_i in zip(X_f, Ra_f):
                X_fold.append(X_i)
                N_fm_fold.append(N_fm)
                N_r_fold.append(N_r)
                Ra_fold.append(Ra_i)

    return X_fold, N_r_fold, N_fm_fold, Ra_fold


def Psi_Plot(Y_FOLD, N_r_FOLD, N_fm_FOLD, axs):
    """Cycle through the fold points and plot them out."""
    count = 0
    for Yi, N_r, N_fm in zip(Y_FOLD, N_r_FOLD, N_fm_FOLD):

        R = cheb_radial(N_r, d)[1]
        Theta_grid = np.linspace(0, np.pi, N_fm)
        r_grid = np.linspace(R[-1], R[0], 50)

        T = Spectral_To_Gridpoints(Yi, R, r_grid, N_fm, d)[1]
        T = T/np.linalg.norm(T, 2)

        axs[count].contour(Theta_grid, r_grid, T, RES, colors='k', linewidths=0.5)
        axs[count].contourf(Theta_grid, r_grid, T, RES, cmap="RdBu_r")
        axs[count].set_xticks([])
        axs[count].set_yticks([])
        # axs[count].set_xlim([0,np.pi])
        #axs[count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)
        axs[count].axis('off')
        count+=1

    return None


def Add_Label(X_folds, Nr_folds, Nfm_folds, Ra_folds, ax):
    """Cycle through the fold points and plot them out."""
    X = []
    Y = []
    for Xi, N_r, N_fm, Ra_i in zip(X_folds, Nr_folds, Nfm_folds, Ra_folds):
        D, R = cheb_radial(N_r, d)
        Ke_i = Kinetic_Energy(Xi, R, D, N_fm, N_r-1, symmetric=False)
        X.append(Ra_i)
        Y.append(Ke_i)

    count = 0
    for xy in zip(X, Y):
        ax.annotate(r'%d' % count, xy=xy, textcoords='data', fontsize=25)
        count += 1

    return None

# %%
print('Load above')
dir = '/home/pmannix/SpectralDoubleDiffusiveConvection/Figure_L10_Snaking/'


# %%

# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 Minus Convectons Ras=450
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convectons_Minus_Ras450/', ax, line='k-')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_xlim([3150, 3450])
ax.set_ylim([1, 7])
# # B) Add other points

# L=10 Plus eigenvector from KE = 0
obj = result()
with h5py.File(dir + 'Convectons_Minus_Ras450/' + "Continuationl10LargeRas450_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 2
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(5, obj.X_DATA[point])
    Ra_folds.insert(5, obj.Ra_DATA[point])
    Nr_folds.insert(5, N_r)
    Nfm_folds.insert(5, N_fm)
    

    point = 3
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(6, obj.X_DATA[point])
    Ra_folds.insert(6, obj.Ra_DATA[point])
    Nr_folds.insert(6, N_r)
    Nfm_folds.insert(6, N_fm)

# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[:, 1])

for count in range(8+1):
    axs[::-1, 1][count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)

# E) Add labels
Add_Label(X_folds[::-1], Nr_folds[::-1], Nfm_folds[::-1], Ra_folds[::-1], ax)

plt.savefig('L10_Minus_Convectons_Ras450.png', format='png', dpi=400)
plt.show()
# %%




# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 minus full Ra_s = 350
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvectons_Minus_Ras350', ax, line='k:')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([1e-02, 5])
ax.set_xlim([3000, 3800])
ax.tick_params(axis='both', labelsize=25)


# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(dir + 'AntiConvectons_Minus_Ras350/' + "Continuationl10MinusTestRa_s350_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 37
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(2, obj.X_DATA[point])
    Ra_folds.insert(2, obj.Ra_DATA[point])
    Nr_folds.insert(2, N_r)
    Nfm_folds.insert(2, N_fm)

    point = 39
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(3, obj.X_DATA[point])
    Ra_folds.insert(3, obj.Ra_DATA[point])
    Nr_folds.insert(3, N_r)
    Nfm_folds.insert(3, N_fm)


    point = 54
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(6, obj.X_DATA[point])
    Ra_folds.insert(6, obj.Ra_DATA[point])
    Nr_folds.insert(6, N_r)
    Nfm_folds.insert(6, N_fm)

    point = 55
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    #ax.semilogy(obj.Ra_DATA[point], KE, 'bs')
    ax.plot(obj.Ra_DATA[point], KE, 'bs')


    X_folds.insert(7, obj.X_DATA[point])
    Ra_folds.insert(7, obj.Ra_DATA[point])
    Nr_folds.insert(7, N_r)
    Nfm_folds.insert(7, N_fm)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[::-1, 1])

for count in range(8+1):
    axs[::-1, 1][count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)


# E) Add labels
Add_Label(X_folds, Nr_folds, Nfm_folds, Ra_folds, ax)

plt.savefig('L10_Minus_AntiConvectons_Ras350.png', format='png', dpi=400)
plt.show()


# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 Plus Anti-convectons Ra_s = 400
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvectons_Plus_Ras400/', ax, line='k:')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([0, 0.75])
ax.set_xlim([3500, 3800])
ax.tick_params(axis='both', labelsize=25)


# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(dir + 'AntiConvectons_Plus_Ras400/' + "AntiConvectonL10PlusRas400_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 17
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)


    point = 18
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(1, obj.X_DATA[point])
    Ra_folds.insert(1, obj.Ra_DATA[point])
    Nr_folds.insert(1, N_r)
    Nfm_folds.insert(1, N_fm)


    point = 32
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(3, obj.X_DATA[point])
    Ra_folds.insert(3, obj.Ra_DATA[point])
    Nr_folds.insert(3, N_r)
    Nfm_folds.insert(3, N_fm)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[::-1, 1])

for count in range(3+1):
    axs[::-1, 1][count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)

# E) Add labels
Add_Label(X_folds, Nr_folds, Nfm_folds, Ra_folds, ax)

plt.savefig('L10_Plus_AntiConvectons_Ras400.png', format='png', dpi=400)
plt.show()



# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 Plus convectons Ra_s = 400
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convectons_Plus_Ras400/', ax, line='k-')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([1, 7])
ax.set_xlim([3100, 3400])
ax.tick_params(axis='both', labelsize=25)


# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(dir + 'Convectons_Plus_Ras400/' + "ConvectonL10PlusRas400_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 30
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(1, obj.X_DATA[point])
    Ra_folds.insert(1, obj.Ra_DATA[point])
    Nr_folds.insert(1, N_r)
    Nfm_folds.insert(1, N_fm)


    point = 31
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(2, obj.X_DATA[point])
    Ra_folds.insert(2, obj.Ra_DATA[point])
    Nr_folds.insert(2, N_r)
    Nfm_folds.insert(2, N_fm)


    point = 46
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(5, obj.X_DATA[point])
    Ra_folds.insert(5, obj.Ra_DATA[point])
    Nr_folds.insert(5, N_r)
    Nfm_folds.insert(5, N_fm)


    point = 47
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(6, obj.X_DATA[point])
    Ra_folds.insert(6, obj.Ra_DATA[point])
    Nr_folds.insert(6, N_r)
    Nfm_folds.insert(6, N_fm)

# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds[:7], Nr_folds[:7], Nfm_folds[:7], axs=axs[:, 1])

for count in range(6+1):
    axs[::-1, 1][count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)

#E) Add labels
Add_Label(X_folds[:7][::-1], Nr_folds[:7][::-1], Nfm_folds[:7][::-1], Ra_folds[:7][::-1], ax)

plt.savefig('L10_Plus_Convectons_Ras400.png', format='png', dpi=400)
plt.show()

# %%
