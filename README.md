# Parrallelised-Spectral-Code-for-Double-Diffusive-Convection
This code approximates the system of partial differential equations, by making use of an expansion and projection onto a basis of N Legendre polynomials. Although accurate this expansion incurs a complexity of ~ N^3, making this code impractical for large N without parallelisation. Using Fortran wrappers and open MPI directives, this code implements time-stepping using a Crank-Nicolson or Euler Implicit scheme.
