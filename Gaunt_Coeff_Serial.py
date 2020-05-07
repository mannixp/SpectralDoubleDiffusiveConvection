#!/usr/bin/env python

## Calculates the weights of \theta dependent coefficients, appearing in nonlinear terms for convection, using Gaunt coefficient formula.

## ~~~~~~ @ CEDRIC @ ~~~~~~~~ 
## Enter the number of Modes here
N_modes = 100+1; ## Set to 200 
N_start = 0;
N_end = 5;
## ~~~~~~~~~~~~~~~~~~~~~~~~~~

import os, sys, time, gc
import numpy as np
import scipy.special as sp
from sympy.physics.wigner import gaunt, wigner_3j

## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
#G = lambda x,l: -sin(x)*sin(x)*gegenbauer(l-1,1.5,cos(x))
#P = lambda x,n: legendre(n,cos(x))

### ~~~~~ Normalising term ~~~~~~~~
def norm_geg(n):
	alpha = 1.5
	return ( ( float(sp.factorial(n))*(n + alpha)*(sp.gamma(alpha)**2) )/( np.pi*(2.0**(1.0-2.0*alpha))*sp.gamma(n +2.0*alpha) ) )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~			
# Returns PSI Coefficients of Gegenbauer Polynomials;
def aPSI(l,m,n,G_0m11): 
	#beta = float(mpmath.quad(lambda x: ( (l*(l+1.0)*P(x,l)*G(x,m))/sin(x))*G(x,n),[0,float(pi)], method='tanh-sinh', maxdegree=20)	)
	#beta = -2.0*l*(l+1.0)*( ( (n*(n+1.0))*(m*(m+1.0)) )**0.5)*float(wigner_3j(l,m,n,0,-1,1)*wigner_3j(l,m,n,0,0,0) )
	beta = -2.0*l*(l+1.0)*( ( (n*(n+1.0))*(m*(m+1.0)) )**0.5)*G_0m11
	
	return beta;
	
def bPSI(l,m,n,G_m101,G_m1m12,G_1m10): 
	# float(mpmath.quad(lambda x: -( m*(m+1.0)*P(x,m) + (2.0*cos(x)*G(x,m))/(sin(x)**2) )*G(x,l)*(G(x,n)/sin(x)),[0,float(pi)], method='tanh-sinh', maxdegree=20) )
	#beta1 = -2.0*m*(m+1.0)*( ( (n*(n+1.0))*(l*(l+1.0)) )**0.5)*float(-wigner_3j(l,m,n,-1,0,1)*wigner_3j(l,m,n,0,0,0))
	#beta2 = ( ( (m*l*n)*(m+1.0)*(l+1.0)*(n+2.0)*(n**2 - 1.0) )**0.5)*float(wigner_3j(l,m,n,-1,-1,2)*wigner_3j(l,m,n,0,0,0));
	#beta3 = - n*(n+1.0)*( ( (l*(l+1.0))*(m*(m+1.0)) )**0.5)*float(wigner_3j(l,m,n,1,-1,0)*wigner_3j(l,m,n,0,0,0));

 	beta1 = 2.0*m*(m+1.0)*( ( (n*(n+1.0))*(l*(l+1.0)) )**0.5)*G_m101
	beta2 = ( ( (m*l*n)*(m+1.0)*(l+1.0)*(n+2.0)*(n**2 - 1.0) )**0.5)*G_m1m12
	beta3 = -n*(n+1.0)*( ( (l*(l+1.0))*(m*(m+1.0)) )**0.5)*G_1m10

	beta = beta1 + 2.0*(beta2 + beta3);

	return beta;


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~			
# Returns Temperature Coefficients of Legendre Polynomials;
def aT(l,m,n,G_000):
	#alpha = 0.5*(2.0*n + 1.0)*float(mpmath.quad(lambda x: l*(l+1.0)*P(x,l)*P(x,m)*P(x,n)*sin(x),[0,float(pi)], method='tanh-sinh',maxdegree=20))
	#alpha =	(2.0*n + 1.0)*l*(l+1.0)*( float(wigner_3j(l,m,n,0,0,0))**2 );

	alpha =	(2.0*n + 1.0)*l*(l+1.0)*G_000
	
	return alpha;

def bT(l,m,n,G_1m10):
	#alpha = 0.5*(2.0*n + 1.0)*float(mpmath.quad(lambda x:((G(x,l)*G(x,m))/sin(x))*P(x,n), [0,float(pi)], method='tanh-sinh',maxdegree=20 ))
	#alpha = -(2.0*n + 1.0)*( ( (l*(l+1.0))*(m*(m+1.0)) )**0.5)*float(wigner_3j(l,m,n,1,-1,0)*wigner_3j(l,m,n,0,0,0)) # 

	alpha = -(2.0*n + 1.0)*( ( (l*(l+1.0))*(m*(m+1.0)) )**0.5)*G_1m10
	
	return alpha;

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

digits = 64; ## Controls the numerical precision of coefficients

# Allocate Arrays, for N_modes = 200 ~ 256mB
#s = (N_modes,N_modes,N_modes);
#aPsi = np.zeros(s); bPsi = np.zeros(s);
#aTt = np.zeros(s); bTt = np.zeros(s);


## Serial Calculion of Coefficients
#@profile
def Calc_Coeffs(N_modes,n_start,n_end):


	n_range = n_end - n_start;

	s = (N_modes,N_modes,n_range);
	aPsi = np.zeros(s); bPsi = np.zeros(s);
	aTt = np.zeros(s); bTt = np.zeros(s);

	for n in xrange(n_range):
		n = n_start + n
		print "Mode l = ",n,"\n"
		norm_psi = norm_geg(n-1)	
		for l in xrange(N_modes):

			for m in xrange(N_modes):

					if (l+m+n)%2 == 0:
		
						if (l <= (m+n)) and (m <= (l+n)) and (n <= (l+m)): 
							
							# Use Additional Symmetries 

							# Calc as common
							norm_g = np.sqrt( (4.0*np.pi)/((2*l + 1)*(2*m+1)*(2*n+1)))

							# Calculate 5 Gaunt coefficients
							m_1, m_2, m_3 = 0,-1,1; G_0m11 = gaunt(l,m,n,m_1, m_2, m_3,prec=digits)*norm_g;
							m_1, m_2, m_3 = -1,0,1; G_m101 = gaunt(l,m,n,m_1, m_2, m_3,prec=digits)*norm_g;
							m_1, m_2, m_3 = -1,-1,2; G_m1m12 = gaunt(l,m,n,m_1, m_2, m_3,prec=digits)*norm_g;
							
							m_1, m_2, m_3 = 1,-1,0; G_1m10 = gaunt(l,m,n,m_1, m_2, m_3,prec=digits)*norm_g;
							m_1, m_2, m_3 = 0,0,0; G_000 = gaunt(l,m,n,m_1, m_2, m_3,prec=digits)*norm_g;

							
							# Determine PSI coefficients	
							aPsi[l,m,n-n_start] = aPSI(l,m,n,G_0m11)*norm_psi; bPsi[l,m,n-n_start] = bPSI(l,m,n,G_m101,G_m1m12,G_1m10)*norm_psi;

							# Determine Temperature Coefficients
							aTt[l,m,n-n_start] = aT(l,m,n,G_000); bTt[l,m,n-n_start] = bT(l,m,n,G_1m10);
							
							#clear cache	
							#gc.collect() # Works but super slow 5 -> 10 times 


	# ~~~~~~ @ CEDRIC ~~~~ 
	# Folder is created locally in execution directory, within which the coefficient arrays are saved
	#~~~~~~~~~~~~~~~~~~~~~~

	Stringy = ['Integrated_Modes',str(N_modes),'_Range',str(n_start),'_to_',str(n_end-1)]

	os.mkdir("".join(Stringy))
	os.chdir("".join(Stringy))

	# Save Psi Coefficients
	np.save("aPsi.npy",aPsi); np.save("bPsi.npy",bPsi);
	# Save T,C Coefficients
	np.save("aT.npy",aTt); np.save("bT.npy",bTt);

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))


###~~~~~~~~~~~ Serial Calculation ~~~~~~~~~~~~~~~~~~~~~~~~~##

start_time = time.time()
Calc_Coeffs(N_modes,N_start,N_end)
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))	