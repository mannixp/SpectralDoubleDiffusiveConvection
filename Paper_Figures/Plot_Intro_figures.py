"""
Script that generates the figures for section 3

To run this script excute

python3 Plot_Intro_figures.py

from within the Paper_Figures directory.
"""

import numpy as np
import glob
import h5py

import sys
import os

sys.path.append(os.path.abspath("../"))

from Matrix_Operators import cheb_radial
from Plot_Tools import Spectral_To_Gridpoints

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

RES = 25  # number of contour levels to use
print('Load above')
dir = '/home/pmannix/Spatial_Localisation/SpectralDoubleDiffusiveConvection/Paper_Data/Intro_Figures/'


# %%

def file_to_mats(filename, frame, Include_Base_State=True):
    if filename.endswith('.h5'):
        f    = h5py.File(filename, 'r');
        X    = f['Checkpoints/X_DATA'][frame,:];
        N_fm = f['Parameters']["N_fm"][()];
        N_r  = f['Parameters']["N_r"][()];
        d    = f['Parameters']["d"][()];
        f.close()

    D, R = cheb_radial(N_r, d)
    Theta_grid = np.linspace(0, np.pi, N_fm)  
    r_grid = np.linspace(R[0], R[-1], 50)

    PSI, Θ, Σ, T_0 = Spectral_To_Gridpoints(X, R,r_grid,N_fm,d)

    if Include_Base_State is True:
        Σ = Σ + T_0
    return Theta_grid, r_grid, Σ

# %%

d = 0.3521
files = glob.glob(dir + '/Even_Parity/*.h5')
print(files[:])
# Make the plot
fig, ax = plt.subplots(1, len(files), subplot_kw=dict(projection='polar'), figsize=(18, 4), layout='constrained')  	
#for file, ax_i, frame, label in zip(files, [ax[2], ax[0], ax[1], ax[3]], [20, 35, 28, 40], [r'(c)', r'(a)', r'(b)', r'(d)']):
for file, ax_i, frame, label in zip(files, [ax[0], ax[1], ax[3], ax[2]], [25, 28, 20, 40], [r'(a)', r'(b)', r'(d)', r'(c)']):
    ax_i.set_theta_zero_location("S")
    ax_i.set_ylim(0, (1+d)/d)
    ax_i.set_rgrids([])
    ax_i.set_thetagrids([0, 90, 180, 270])
    ax_i.set_xticklabels({r'S':0, r'E':90, r'N':180, r'W':270}, fontsize=20)


    Theta_grid, r_grid, Σ = file_to_mats(file, frame)
    ax_i.contourf(Theta_grid, r_grid, Σ, RES, cmap="RdBu_r")
    ax_i.contourf(2.0*np.pi-Theta_grid, r_grid, Σ, RES, cmap="RdBu_r")
    #ax_i.contour(Theta_grid, r_grid, Σ, RES, colors='k')
    #ax_i.contour(2.0*np.pi-Theta_grid, r_grid, Σ, RES, colors='k')
    ax_i.annotate(label, xy=(-0.05, 0.95), xycoords='axes fraction', fontsize=20)


plt.savefig('Localised_Solutions_Even_Plot.png', format='png', dpi=100)
plt.show()

# %%

d = 0.31325
files = glob.glob(dir + '/Odd_Parity/*.h5')
print(files)
# Make the plot
fig, ax = plt.subplots(1, len(files), subplot_kw=dict(projection='polar'), figsize=(9, 4), layout='constrained')  	
for file, ax_i, frame, label in zip(files, ax, [5, -1], [r'(a)', r'(b)']):
    ax_i.set_theta_zero_location("S")
    ax_i.set_ylim(0, (1+d)/d)
    #ax_i.axis("off")

    ax_i.set_rgrids([])
    ax_i.set_thetagrids([0, 90, 180, 270])
    ax_i.set_xticklabels({r'S':0, r'E':90, r'N':180, r'W':270}, fontsize=20)

    Theta_grid, r_grid, Σ = file_to_mats(file, frame)
    ax_i.contourf(Theta_grid, r_grid, Σ, RES, cmap="RdBu_r")
    ax_i.contourf(2.0*np.pi-Theta_grid, r_grid, Σ, RES, cmap="RdBu_r")
    #ax_i.contour(Theta_grid, r_grid, Σ, RES, colors='k')
    #ax_i.contour(2.0*np.pi-Theta_grid, r_grid, Σ, RES, colors='k')
    ax_i.annotate(label, xy=(-0.05, 0.95), xycoords='axes fraction', fontsize=20)


plt.savefig('Localised_Solutions_Odd_Plot.png', format='png', dpi=100)
plt.show()
# %%
