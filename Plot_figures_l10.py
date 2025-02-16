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

        ax.semilogy(obj.Ra, obj.KE, line)
        ax.semilogy(Ra_f, KE_f, 'ro', markersize=markersize)

        #ax.plot(obj.Ra, obj.KE, line)
        #ax.plot(Ra_f, KE_f, 'ro', markersize=markersize)


        # Return Saddles
        if len(obj.Y_FOLD) != 0:
            return obj.Y_FOLD[idx]
        else: 
            return []
        
    X_fold = []
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

            X_f = add_to_fig(obj)
            for i, X_i in enumerate(X_f):
                X_fold.append(X_i)
                N_fm_fold.append(N_fm)
                N_r_fold.append(N_r)

    return X_fold, N_r_fold, N_fm_fold


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



# %%
print('Load above')
dir = '/home/pmannix/SpectralDoubleDiffusiveConvection/Branches_l10_d0.3521/Branches/'

# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# Transcritical L = 10 plus & minus
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')

for filename, name in zip( glob.glob(dir + '/Transcritical' + '/*.h5'), [r'$\ell=10^-$', r'$\ell=10^+$'] ):
    obj = result()
    with h5py.File(filename, 'r') as f:
        ff = f["Bifurcation"]
        for key in ff.keys():
            setattr(obj, key, ff[key][()])
        if name == r'$\ell=10^-$':
            ax.plot(obj.Ra, obj.KE, 'k-', linewidth=1.5, label=name)
        else:
            ax.plot(obj.Ra, obj.KE, 'k-.', linewidth=1.5, label=name)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([0, 0.0015])
ax.tick_params(axis='both', labelsize=25)
ax.legend(fontsize=25)

plt.savefig('Bifurcation_L10_Transcritical.png', format='png', dpi=400)
plt.show()


# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 minus
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds = Plot_full_bif(dir + 'Minus/', ax)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([1e-04, 3])
ax.tick_params(axis='both', labelsize=25)

# B) Add other points

# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(dir+"Minus/Continuationl10MinusTest_0.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    idx = np.asarray([i for i, x in enumerate(obj.KE) if x <= 1.1e-04])[-1]
    ax.semilogy(obj.Ra[idx], obj.KE[idx], 'ks', markersize=markersize)
    
    X_folds.insert(0, obj.X_DATA[0])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

# 2-pulse state
point = 10
obj = result()
with h5py.File(dir+"Minus/Continuationl10MinusTest_4.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])

    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1)
    ax.semilogy(obj.Ra_DATA[point], KE, 'ro',  markersize=markersize)
    
    X_folds.insert(4, obj.X_DATA[point])
    Nr_folds.insert(4, N_r)
    Nfm_folds.insert(4, N_fm)

# L=10 branch going back to KE = 0
obj = result()
with h5py.File(dir+"Minus/Continuationl10MinusTest_4.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    idx = np.asarray([i for i, x in enumerate(obj.KE) if x > 0.9e-04])[-1]
    ax.semilogy(obj.Ra[idx], obj.KE[idx], 'ro',  markersize=markersize)
    
    # Not added because a fold point with the same state has already been added
    # X_folds.insert(5, obj.X_DATA[-1])
    # Nr_folds.insert(5, N_r)
    # Nfm_folds.insert(5, N_fm)


# C) Create the inset
axins = inset_axes(ax, width="30%", height="40%", loc='upper right', borderpad=1)
Plot_full_bif(dir + 'Minus/', axins)

axins.semilogy(obj.Ra_DATA[point], KE, 'ro', markersize=markersize) # Add the 2-pulse state to the inset

axins.set_ylim([5e-02, 3])
axins.set_xlim([3500, 4250])
axins.tick_params(axis='both', labelsize=25)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds[:6], Nr_folds[:6], Nfm_folds[:6], axs=axs[::-1, 1])

plt.savefig('Bifurcation_L10_Minus.png', format='png', dpi=400)
plt.show()







# %%

# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 Plus
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds = Plot_full_bif(dir + 'Plus/', ax)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([1e-04, 3])
ax.tick_params(axis='both', labelsize=25)

# B) Add other points

# L=10 Plus eigenvector from KE = 0
obj = result()
with h5py.File(dir+"Plus/Continuationl10PlusTest_0.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    idx = np.asarray([i for i, x in enumerate(obj.KE) if x <= 1.1e-04])[-1]
    ax.semilogy(obj.Ra[idx], obj.KE[idx], 'ks', markersize=markersize)
    
    X_folds.insert(0, obj.X_DATA[0])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

# 2-pulse state
point = 130
obj = result()
with h5py.File(dir+"Plus/Continuationl10PlusTest_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])

    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1)
    ax.semilogy(obj.Ra_DATA[point], KE, 'ro',  markersize=markersize)

    X_folds.insert(4, obj.X_DATA[point])
    Nr_folds.insert(4, N_r)
    Nfm_folds.insert(4, N_fm)

# L=10 Plus branch going back to KE = 0
obj = result()
with h5py.File(dir+"Plus/Continuationl10PlusTest_15.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    idx = np.asarray([i for i, x in enumerate(obj.KE) if x > 7.5e-05])[-1]
    ax.semilogy(obj.Ra[idx], obj.KE[idx], 'ro',  markersize=markersize)
    
    X_folds.insert(6, obj.X_DATA[-1])
    Nr_folds.insert(6, N_r)
    Nfm_folds.insert(6, N_fm)

# C) Create the inset

axins = inset_axes(ax, width="30%", height="20%", loc='upper right', borderpad=2)
Plot_full_bif(dir + 'Plus/', axins)

axins.semilogy(obj.Ra_DATA[point], KE, 'ro', markersize=markersize) # Add the 2-pulse state to the inset

axins.set_ylim([1e-01, 1])
axins.set_xlim([3700, 3900])
axins.tick_params(axis='both', labelsize=25)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds[:7], Nr_folds[:7], Nfm_folds[:7], axs=axs[::-1, 1])

plt.savefig('Bifurcation_L10_Plus.png', format='png', dpi=400)
plt.show()






# %%

# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 Large
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds = Plot_full_bif(dir + 'Large/', ax)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)


# B) Add other points

# L=10 Plus eigenvector from KE = 0
obj = result()
with h5py.File(dir+"Large/Continuationl10Large_a.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    ax.semilogy(obj.Ra[-1], obj.KE[-1], 'ks', markersize=markersize)
    
    X_folds.insert(0, obj.X_DATA[-1])
    Nr_folds.insert(0, N_r)
    Nfm_folds.insert(0, N_fm)

# L=10 Plus branch going back to KE = 0
obj = result()
with h5py.File(dir+"Large/Continuationl10Large_17.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    ax.semilogy(obj.Ra[-1], obj.KE[-1], 'ro',  markersize=markersize)
    
    X_folds.insert(8, obj.X_DATA[-1])
    Nr_folds.insert(8, N_r)
    Nfm_folds.insert(8, N_fm)


# C) Add the inset
axins = inset_axes(ax, width="70%", height="60%", 
                   bbox_to_anchor=(0.35, 0.35, .6, .5),
                   bbox_transform=ax.transAxes,
                   loc='upper right', borderpad=2)
Plot_full_bif(dir + 'Large/', axins)
axins.set_ylim([2, 10])
axins.set_xlim([3200, 3550])
axins.tick_params(axis='both', labelsize=25)

# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds[:9], Nr_folds[:9], Nfm_folds[:9], axs=axs[::-1, 1])


plt.savefig('Bifurcation_L10_Large.png', format='png', dpi=400)
plt.show()




# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 minus full Ra_s = 350 Tau = Tau/2
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

dir = '/home/pmannix/SpectralDoubleDiffusiveConvection/Branches_l10_d0.353/'
folder = dir + 'Branch_l10_Minus_d0.353_Ra_s350_0.5Tau/'


fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds = Plot_full_bif(folder, ax)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([1e-02, 5])
ax.set_xlim([3000, 4000])
ax.tick_params(axis='both', labelsize=25)


# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(folder + "ContinuationMinusRas350_1.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]

    D, R = cheb_radial(N_r, d)
    
    point = 37
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(2, obj.X_DATA[point])
    Nr_folds.insert(2, N_r)
    Nfm_folds.insert(2, N_fm)

    point = 39
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(3, obj.X_DATA[point])
    Nr_folds.insert(3, N_r)
    Nfm_folds.insert(3, N_fm)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[::-1, 1])

plt.savefig('Bifurcation_L10_Minus_Ras350_0.5Tau.png', format='png', dpi=400)
plt.show()


# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 10 minus full Ra_s = 350
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~


dir = '/home/pmannix/SpectralDoubleDiffusiveConvection/Branches_l10_d0.353/'
folder = dir + 'Branch_l10_Minus_d0.353_Ra_s350/'

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(16, 6), layout='constrained')

# remove the underlying Axes in the right column
gs = axs[1, 0].get_gridspec()
for axl in axs[:, 0]:
    axl.remove()
ax = fig.add_subplot(gs[:, 0])

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds = Plot_full_bif(folder, ax)

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([2e-02, 2e01])
ax.set_xlim([3000, 4000])
ax.tick_params(axis='both', labelsize=25)


# L=10 Minus eigenvector from KE = 0
obj = result()
with h5py.File(folder + "ContinuationMinusRas350_0.h5", 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])
    N_fm = f['Parameters']["N_fm"][()]
    N_r = f['Parameters']["N_r"][()]
    
    D, R = cheb_radial(N_r, d)
    
    point = 33
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(2, obj.X_DATA[point])
    Nr_folds.insert(2, N_r)
    Nfm_folds.insert(2, N_fm)

    point = 35
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(3, obj.X_DATA[point])
    Nr_folds.insert(3, N_r)
    Nfm_folds.insert(3, N_fm)


    point = 51
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(6, obj.X_DATA[point])
    Nr_folds.insert(6, N_r)
    Nfm_folds.insert(6, N_fm)

    point = 52
    KE = Kinetic_Energy(obj.X_DATA[point], R, D, N_fm, N_r-1, symmetric=False)
    ax.semilogy(obj.Ra_DATA[point], KE, 'bs')

    X_folds.insert(7, obj.X_DATA[point])
    Nr_folds.insert(7, N_r)
    Nfm_folds.insert(7, N_fm)


# D) Plot the points out in terms of their stream-function
Psi_Plot(X_folds, Nr_folds, Nfm_folds, axs=axs[::-1, 1])

plt.savefig('Bifurcation_L10_Minus_Ras350.png', format='png', dpi=400)
plt.show()

# %%
