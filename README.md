# SphericalDoubleDiffusiveConvection

This code solves the system of partial differential equations

$$
\begin{align} 
\frac{\partial \boldsymbol{u}}{\partial t} + \left( \boldsymbol{u} \cdot \nabla  \right) \boldsymbol{u}  &= - \nabla P + g(r) Pr \hat{\boldsymbol{r}} (Ra_T T  -  Ra_S S)  + Pr \nabla^2 \boldsymbol{u},\\
\frac{\partial T}{\partial t} + \boldsymbol{u} \cdot \nabla T  &= \nabla^2 T,\\
\frac{\partial S}{\partial t} + \boldsymbol{u} \cdot \nabla S  &= \nabla^2 S,\\
\nabla \cdot \boldsymbol{u} &= 0,\\
\end{align} 
$$

by expanding the fields $\boldsymbol{u},T,S$ in terms of a finite number $N_{\theta}$ Sine\Cosine modes in the latitudinal direction and $N_r$ Chebyshev collocation points in the radial direction. In order to evaluate the nonlinear terms a psuedospectral method is used, whereby the discrete sine/cosine transforms are used to transform the fields from real grid space where nonlinear terms can be evaluated. The inverse sine/cosine transforms are then used to reover the projection of the nonlinear term in spectral space. Using this discretisation the following algorithims are implemented:

**Linear Stability**

This routine solves the system above linearised about the conductive base state $\boldsymbol{u}=0,T = T_0(r),S = S_0(r)$. It can be run by activating an Anaconda environment and then executing 

`python3 Linear_Problem.py`

This computes the eigenvalues/eigenvectors for the parameters selected and can also compute the neutral stability curves over a given parameter range.

**Time integration**

Starting from a random initial condition (default) or a precomputed solution such as a eigenvector or steady state this routine time-integrates the full system of nonlinear equations using a Crank-Nicolson scheme for the linear terms and an Adams-Bashforth scheme for the nonlinear terms. It can be run by activating an Anaconda environment and then executing 

'python3 Time_Continuation.py'

This time-integrates the equations over the time-interval specified for a given set of parameters. The output stored in a `.h5py` file can then be plotted using the `Plot_Tools.py` module.

**Steady state solver (Netwon-iteration)**

In order to compute steady-state solutions of the system of equations, a matrix free Netwon-iteration routine is implemented. Starting from an initial condition which is "close" to the solution for a given set of parameters this routine solves the full system of nonlinear equations. It can be run by activating an Anaconda environment and then executing 

'python3 Main.py'

The output stored in a `.h5py` file can then be plotted using the `Plot_Tools.py` module.

**Arc length continuation**

In order to follow branches of solutions to understand how they depend on system parameters such as $Ra_T$, a psuedo-arc length continuation routine is implemented. Specifying an initial condition and a direction i.e. increasing or decreasing $Ra_T$ this routine follows the full system of equations for a given number of points on the solution branch. It can be run by activating an Anaconda environment and then executing 

'python3 Main.py'

The output stored in a `.h5py` file can then be plotted using the `Plot_Tools.py` module.

## Citation

Please cite the following paper in any publication where you find the present codes useful:

Paul M. Mannix & Cedric Beaume. Spatially localised double-diffusive convection in an axisymmetric spherical shell, JFM, 2025.

```
@article{Mannix_2024,
   author = {Paul M. Mannix and Cedric Beaume},
   title = {Spatially localised double-diffusive convection in an axisymmetric spherical shell},
   journal = {Journal of fluid dynamics},
   pages = {},
   publisher = {Cambridge University Press},
   volume = {},
   year = {2025}
}
```
