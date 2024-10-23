"""
Script that runs verification tests & checks
"""
import numpy as np
from Matrix_Operators import cheb_radial
from Main import Build_Matrix_Operators_TimeStep
from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP
from Matrix_Operators import A4_BSub_TSTEP_V2, NAB2_BSub_TSTEP_V2
from Matrix_Operators import NAB2_TSTEP_MATS, A4_TSTEP_MATS

# Parameters
d = 1
N_fm = 16
N_r = 8
dt = 10**4
symmetric = True

D, R = cheb_radial(N_r, d)
nr = len(R[1:-1])
g = np.random.randn(N_fm*nr)

# (1) Compute the answer
args_Nab2, args_A4 = Build_Matrix_Operators_TimeStep(N_fm, N_r, d)[-3:-1]

f_T = NAB2_BSub_TSTEP(g, *args_Nab2, dt, symmetric)
f_ψ = A4_BSub_TSTEP(g, *args_A4, dt, symmetric)

# (2) Test the new code
L_inv_NAB2 = NAB2_TSTEP_MATS(dt, N_fm, nr, D, R)
f_T_test = NAB2_BSub_TSTEP_V2(g, L_inv_NAB2, N_fm, nr, dt, symmetric)

L_inv_A4 = A4_TSTEP_MATS(dt, N_fm, nr, D, R)
f_ψ_test = A4_BSub_TSTEP_V2(g,  L_inv_A4, D, R, N_fm, nr, dt, symmetric)

# (3) Check
print('Nabla^2 operator = ', np.linalg.norm(f_T - f_T_test, 2), '\n')
print('A^4 operator = ', np.linlag.norm(f_ψ - f_ψ_test, 2), '\n')
