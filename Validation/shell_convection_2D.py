"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.py` script can be used to produce plots from the
saved data. The simulation should take about 20 cpu-minutes to run.

The problem is non-dimensionalized using the shell thickness $d = r_2 - r_1$
and diffusion time d^2/kappa, so the Prandtl and Rayleigh numbers are:

    Rayleigh = 
    Prandtl  =

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shell_convection_2D.py
    $ mpiexec -n 4 python3 plot_shell.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Nphi, Ntheta, Nr = 4, 48, 24  # 4 modes is minimum resolution

# l = 2
Ra = 6780.0; d = 2; Pr=10;
#Ra = 2360.0; d = 0.353; Pr=1;

# l = 11
#Ra = 2280; d = 0.31325; Pr=1;  # Must enforce symmetry 

Ri, Ro = 1./d, (1. + d)/d;
dealias  = 3/2
stop_sim_time = 2*(10**3)
timestepper = d3.SBDF2
max_timestep = 1
dtype = np.float64
mesh = None
Ri, Ro = 1./d, (1. + d)/d;

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
sphere = shell.outer_surface

# Fields
p = dist.Field(name='p', bases=shell)
b = dist.Field(name='b', bases=shell)
u = dist.VectorField(coords, name='u', bases=shell)
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=sphere)
tau_b2 = dist.Field(name='tau_b2', bases=sphere)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)

# Substitutions
phi, theta, r = dist.local_grids(shell)
er = dist.VectorField(coords, bases=shell.radial_basis)
er['g'][2] =  1

gr = dist.VectorField(coords, bases=shell.radial_basis)
gr['g'][2] = (Ri/r)**2

rvec = dist.VectorField(coords, bases=shell.radial_basis)
rvec['g'][2] = r

lift_basis = shell.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - div(grad_b)                     + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - div(grad_u) + grad(p) - gr*Ra*b + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(r=Ri) = 1")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("b(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
b['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # Add linear background

# Analysis
flux = er @ (-d3.grad(b) + u*b)
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10)
snapshots.add_task(b(phi=0), scales=dealias, name='b(phi=0)')

snapshots.add_task(b(r=(Ri+Ro)/2), scales=dealias, name='bmid')
snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')
snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')
snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')
snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')


# Check how d3.Integrate works on (phi, theta, r)
f = dist.Field(name='f',bases=shell)
f['g'] = 1;
f_int = d3.Integrate(f)
V = f_int.evaluate()['g'][0,0,0]
# Yields # 4/3*pi*(Ro^3 - Ri^3) 

db0 = dist.Field(name='db0',bases=shell);
A_T = (Ri*Ro)/(Ri - Ro)
db0['g'] = A_T/(r**2);
dbi_int  = d3.Average(db0(r=Ri),coords.S2coordsys).evaluate()['g'][0,0,0]
dbo_int  = d3.Average(db0(r=Ro),coords.S2coordsys).evaluate()['g'][0,0,0]

# Scalar data
scalar = solver.evaluator.add_file_handler('scalar_data', iter=10)
scalar.add_task((.5/V)*d3.Integrate(u@u),layout='g',name='(1/2)<u^2>')
scalar.add_task(d3.Average(er@d3.grad(b)(r=Ri),coords.S2coordsys)/dbi_int, name='Nu')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((.5/V)*d3.Integrate(u@u), name='(1/2)<u^2>')
flow.add_property(d3.Average(er@d3.grad(b)(r=Ri),coords.S2coordsys)/dbi_int, name='Nu_i')
flow.add_property(d3.Average(er@d3.grad(b)(r=Ro),coords.S2coordsys)/dbo_int, name='Nu_o')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        
        # Enforce axisymmetry
        b['g'][:, :, :] = b['g'][0, :, :]
        u['g'][0, :, :, :] = u['g'][0, 0, :, :]
        u['g'][1, :, :, :] = u['g'][1, 0, :, :]
        u['g'][2, :, :, :] = u['g'][2, 0, :, :]

        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            KE = flow.grid_average('(1/2)<u^2>')
            Nu = flow.max('Nu_i')-1.
            Nu_diff = abs(flow.max('Nu_o') - flow.max('Nu_i') )

            logger.info('Iteration=%i, Time=%e, dt=%e, (1/2)<u^2>=%f, Nu-1=%e' %(solver.iteration, solver.sim_time, timestep, KE, Nu))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()