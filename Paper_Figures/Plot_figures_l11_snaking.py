"""
Script that generates the figure 15 for section 5.2

To run this script excute

python3 Plot_figures_l11_snaking.py

from within the Paper_Figures directory.
"""
import numpy as np
import glob, h5py

import sys
import os

sys.path.append(os.path.abspath("../"))

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


d = 0.31325  # Fixed gap-width
RES = 25  # number of contour levels to use
markersize = 5


def Plot_full_bif(folder, ax, line='k-', label='ro'):
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

        ax.plot(obj.Ra, obj.KE, line)
        ax.plot(Ra_f, KE_f, label, markersize=markersize)

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


# %%
print('Load above')
dir = '/home/pmannix/Spatial_Localisation/SpectralDoubleDiffusiveConvection/Paper_Data/Figure_L11/'


# %%
# L = 11 Ras=150
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(16, 8), layout='constrained')

# remove the underlying Axes in the middle column
for axl in axs[:, 1]:
    axl.remove()

# Add a single plot to it
gs = axs[1, 0].get_gridspec()
ax = fig.add_subplot(gs[:, 1])


# A) Plus Branch
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
label='bs'
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Anti_Convectons_Plus/' , ax, line='k-.')

obj = result()
with h5py.File(dir + 'Anti_Convectons_Plus/' + "Continuationl11Ras150_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)

    point = 59
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(4, obj.X_DATA[point])
    Ra_folds.insert(4, obj.Ra_DATA[point])
    Nr_folds.insert(4, N_r)
    Nfm_folds.insert(4, N_fm)

    point = 58
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(4, obj.X_DATA[point])
    Ra_folds.insert(4, obj.Ra_DATA[point])
    Nr_folds.insert(4, N_r)
    Nfm_folds.insert(4, N_fm)

    point = 33
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)
    
    point = 32
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

    point = 30
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

Psi_Plot(X_folds[::-1][:-1], Nr_folds[::-1][:-1], Nfm_folds[::-1][:-1], axs=axs[:, 0])

for count in range(9+1):
    axs[::-1, 0][count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)

# E) Add labels
X = []
Y = []
for Xi, N_r, N_fm, Ra_i in zip(X_folds[:-1], Nr_folds[:-1], Nfm_folds[:-1], Ra_folds[:-1]):
    D, R = cheb_radial(N_r, d)
    Ke_i = Kinetic_Energy(Xi, R, D, N_fm, N_r-1, symmetric=False)
    X.append(Ra_i+5)
    Y.append(Ke_i)

count = 0
for xy in zip(X, Y):
    ax.annotate(r'%d' % count, xy=xy, textcoords='data', fontsize=20)
    count += 1

# B) Minus Branch
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
label = 'bs'
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Anti_Convectons_Minus/', ax, line='k-')

# # B) Add other points
obj = result()
with h5py.File(dir + 'Anti_Convectons_Minus/' + "Continuationl11Ras150_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = -14
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

    point = -15
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(0, obj.X_DATA[point])
    Ra_folds.insert(0, obj.Ra_DATA[point])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)


# # B) Add other points
obj = result()
with h5py.File(dir + 'Anti_Convectons_Minus/' + "Continuationl11Ras150_3.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 6
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.plot(obj.Ra_DATA[point], KE, label,markersize=markersize)

    X_folds.insert(8, obj.X_DATA[point])
    Ra_folds.insert(8, obj.Ra_DATA[point])
    Nr_folds.insert(8, N_r)
    Nfm_folds.insert(8, N_fm)

Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[::-1, 2])

letters = [r'a',r'b',r'c',r'd',r'e',r'f',r'g',r'h',r'i',r'j']
for count, letter in enumerate(letters):
    axs[::-1, 2][count].annotate(letter, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20,color='g')


# E) Add labels
X = []
Y = []
for Xi, N_r, N_fm, Ra_i in zip(X_folds[:-1], Nr_folds[:-1], Nfm_folds[:-1], Ra_folds[:-1]):
    D, R = cheb_radial(N_r, d)
    Ke_i = Kinetic_Energy(Xi, R, D, N_fm, N_r-1, symmetric=False)
    X.append(Ra_i-15)
    Y.append(Ke_i-0.075)

count = 0
for xy in zip(X, Y):
    ax.annotate(letters[count], xy=xy, textcoords='data', fontsize=20, color='g')
    count += 1

# Lable & limit the middle axis
ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra_T$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_xlim([2660, 3000])
ax.set_ylim([0, 3.25])

plt.savefig('L11_Snaking_Ras150.png', format='png', dpi=100)
plt.show()