"""
Script that generates the figure 7 for section 5.1

To run this script excute

python3 Plot_figures_L10_bifurcation.py

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

        ax.plot(obj.Ra, obj.KE, line)

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
        axs[count].annotate(r'%d' % count, xy=(-0.05, 0.5), xycoords='axes fraction', fontsize=20)
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
        ax.annotate(r'%d' % count, xy=xy, textcoords='data', fontsize=20)
        count += 1

    return None

# %%
print('Load above')
dir = '/home/pmannix/Spatial_Localisation/SpectralDoubleDiffusiveConvection/Paper_Data/Figure_L10_Full_Bif_Ras400/'


# %%

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), layout='constrained', sharey=True)

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convecton_L10_Plus/', ax[0], line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvecton_L10_Plus/', ax[0], line='k:')

ax[0].set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax[0].set_xlabel(r'$Ra_T$', fontsize=25)
ax[0].tick_params(axis='both', labelsize=25)
#ax[0].set_title(r'$\ell=10^{+}$', fontsize=25)
ax[0].set_ylim([0, 12.5])
ax[0].set_xlim([3000, 8500])
ax[0].annotate('(a)', xy=(0.0, 1.05), xycoords='axes fraction', fontsize=25)

# B) Create the transcritical inset
axins_0 = inset_axes(ax[0], width="70%", height="70%", loc='upper right', borderpad=2)

X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convecton_L10_Plus/', axins_0, line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvecton_L10_Plus/', axins_0, line='k:')

axins_0.annotate(r'$\ell = 10$', xy=(8300,-0.0006), textcoords='data', fontsize=20, rotation =0)
axins_0.annotate(r'$\ell = 8$', xy=(8425,-0.0006), textcoords='data', fontsize=20, rotation =0)
axins_0.annotate(r'$L_{10}^{C+}$', xy=(8125,0.005), textcoords='data', fontsize=25, rotation =0)
axins_0.annotate(r'$L_{10}^{A+}$', xy=(8200,0.0006), textcoords='data', fontsize=25, rotation =0)


axins_0.tick_params(axis='both', labelsize=25)
axins_0.set_ylim([-0.00075, 0.006])
axins_0.set_xlim([8000, 8500])
axins_0.plot(np.arange(8000, 8500), 0*np.arange(8000, 8500), 'k-')


# C) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convecton_L10_Minus/', ax[1], line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvecton_L10_Minus/', ax[1], line='k:')

#ax[1].set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax[1].set_xlabel(r'$Ra_T$', fontsize=25)
ax[1].tick_params(axis='both', labelsize=25)
#ax[1].set_title(r'$\ell=10^{-}$', fontsize=25)
ax[1].set_ylim([0, 12.5])
ax[1].set_xlim([3000, 8500])
ax[1].annotate('(b)', xy=(0.0, 1.05), xycoords='axes fraction', fontsize=25)

# D) Create the transcritical inset

axins_1 = inset_axes(ax[1], width="70%", height="70%", loc='upper right', borderpad=2)

X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'Convecton_L10_Minus/', axins_1, line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'AntiConvecton_L10_Minus/', axins_1, line='k:')

axins_1.annotate(r'$\ell = 10$', xy=(8300,-0.0006), textcoords='data', fontsize=20, rotation =0)
axins_1.annotate(r'$\ell = 12$', xy=(8425,-0.0006), textcoords='data', fontsize=20, rotation =0)
axins_1.annotate(r'$L_{10}^{C-}$', xy=(8180,0.003), textcoords='data', fontsize=25, rotation =0)
axins_1.annotate(r'$L_{10}^{A-}$', xy=(8200,0.0006), textcoords='data', fontsize=25, rotation =0)

axins_1.tick_params(axis='both', labelsize=25)
axins_1.set_ylim([-0.00075, 0.006])
axins_1.set_xlim([8000, 8500])
axins_1.plot(np.arange(8000, 8500), 0*np.arange(8000, 8500), 'k-')

plt.savefig('Bifurcation_L10_Ras400.png', format='png', dpi=100)
#plt.show()