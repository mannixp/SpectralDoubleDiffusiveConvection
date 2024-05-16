# AxisymmetricSphericalDoubleDiffusiveConvection

**Linear Stability Code**

This code approximates the system of linearised partial differential equations, by making use of an expansion and projection onto a basis of $N_{\theta}$ Legendre polynomials and $N_r$ Chebyshev collocation points in the radial direction. Activating an Anaconda environment with the appropriate dependencies installed `numpy, matplotlib, scipy` and then executing 

`python3 Linear_Problem.py`

computes and plots the neutral curves shown in figure 2 of (Mannix et al. JFM 202?).

Computing one of the steady eigenvalue/eigenvector pairs i.e. no imaginary part and passing this to the Newton solver we can obtain a starting point for continuation. To facilitate convergence of the iterative Krylov solver it is best to use a small resolution $N_{\theta},N_r = 50,20$ and a slightly subcritical Rayleigh number given that the branch is subcritical. Scaling the amplitude of the eigenvector may also be a useful trick to aid convergence.

**Time stepping**

**Steady state code**

**Continuation**
