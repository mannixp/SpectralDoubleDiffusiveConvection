This document contains instructions relating to the pseudo-spectral code contained in this folder.

1) Code-Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NOTE !!!!! The only parameter ever changed is Ra,Nr,N_theta,dt, otherwise all are fixed !!!!

Time-stepping, solve M*X_t = L*X + N(X,X), is carried out using:
TSTEP_SYM.py
TSTEP_ASYM.py

Netwon-Iteration & Continuation, solve L*X + N(X,X) = 0, using:
Routines_MFREE_SYM.py
Routines_MFREE_ASYM.py

Sym & Asym (or otherwise) refers to code where no equator symmetry is enforced.

Other codes include the linear solvers for computing eigenvalues and eigenvectors of the conductive base-state, Folders are good for Neuttral curves (Ra vs d), Nu_vs_Ra near onset, plotting eigenfunctions.


In turn, these codes depend on submodules which have optimised routines for computing L*X and N(X,X):

B_Matrix_sd.py mainly treats non-linear terms, as I've not yet succeded in (# TASK )using fft with Fortan!!!

#~~~~~ F2PY_LIN_MULT.f90 ~~~~~~# which generates python module #~~~~ L_PARA ~~~~~ # for matrix multiplications/inverses and radial differentiations # Uses MKL only locally.

#~~~~~ F2PY_TSTEP.f90 ~~~~~~# which generates python module #~~~~ LT_STEP ~~~~~ # for time-stepping matrix inverses. # Currently doesn't use MKL.

# TASK !! For clarity these .f90 files, which are excessively long, should have been split up into sym and asymm but I have not yet done so.


# 1b) Code compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compilation of these .f90 files with f2py to generate a python modulem, can be performed using gnu95 or ifort depending on which is installed. Currently I've problems with gnu95, i think??

Note the following when doing so, 

1) the numpy version used locally and on the cluster/other computer must match exactly!
2) For optimal python code, use numpy compiled with ifort or appropriate. (I haven't yet looked at this)
3) If ifort is used, the appropriate source link must be added on the computing cluster. This amounts to adding the equivalent of: 

source /usr/local/intel/composer_xe_2011_sp1.10.319/bin/compilervars.sh intel64

in the run.sh file, submitted using torque qsub command.

Additional flags or parralelisation can be specified to achieve speed ups when compiling, but first check that these are also installed on the computing cluster.

1) Intel mkl permits further optimised compiling
2) Openmp permits parralelisation but read further

Link, using f2py: https://docs.scipy.org/doc/numpy-1.13.0/f2py/usage.html ###FIX??
Link, using ifort & MKL specifying correct flags and link lines: https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html


2) Results-Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A) Time-Stepping Results contained in folder: ~/Dropbox/LEEDS2/TEMPORAL

To process these run: PLOT_TSTEP.py, by fixing the appropriate directory. Go to other directory Leeds_OLD to find video making code appropriate!!!
Typically, the residual error from time-stepping must be less than 1e-08 from experience.

B) Continuation Results contained in folder: ~/Dropbox/LEEDS2/Results2 if l = 20 is dominant mode, otherwise ~/Dropbox/LEEDS2/Results_l18_l22 if these are the dominant modes.

# TASK this folder should be split up into three different folders but I have not done-so as of yet. Also delete and repair, dud results.

To process these run:

PLOT_BIF_L20P.py # If L20(+) is the dominante mode
PLOT_BIF_L20N.py # If L20(-) is the dominante mode
PLOT_BIF_L21.py # If L21 is the dominante mode

# TASK A big list of directories is currently used, but these need tidying up


# Note on downloading and uploading files from cluster:
1) Use filezilla, as much faster.
2) Turn off Dropbox sync until end of day
3) Use a differnt port to 22 specify using -p 10022 flag after ssh for example.

#Note-on interpolation between different grid-sizes
1) Radial grid easily interpolated between
2) Polar grid must be done, in intermediate stages/steps, adjusting epsilon_lgmres to be lower helps.

# Note on initiating Newton-Iteration from eigenvectors
1) Keep the resolution low, i.e. Nr = 15 x N_theta = 100, as this greatly facilitates convergence.

3) Additional Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Bug in my continuation appears to cause jumping, needs fixing
# I require to be able to jump Isolas needs fixing: Add deflation routine.
# Changes to Nr, cause a greater slow-down than do, N_theta.
# As the grid size increases for large amplitude space GMRES requires a larger Krylov subspace.


# Profiling code

kernprof -l -v file.py # is currently used to profile python code
# TASK profile fortran code???

# Benchmarking and unit-testing ask about these!!!!S

