from numba import njit
import numpy as np
import sys

from Transforms import IDCT,DCT,IDST,DST

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def cheb_radial(N,d):
	
	r_i = 1.0/d; 
	r_o = (1.0+d)/d;

	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		x = 0.5*(r_i + r_o) + 0.5*(r_i-r_o)*x; # Transform to radial

		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	
	return D, x.reshape(N+1);

# ~~~~~~~~~~~~ Main Linear Operators ~~~~~~~~~~

def Nabla2(D,r): 

	"""
	Build Operator - ∆ pre-mulitplied by r^2

	"""

	# r^2 T'' +2r T'
	D2 = np.diag(r[:]**2)@(D@D);
	RD = np.diag(2.0*r[:])@D;
	A  = D2 + RD

	# Leaving out the edges enforces the dircihlet b.c.s
	return A[1:-1,1:-1];

def Nabla4(D,r):
	
	"""
	
	Build Operator - ∆∆ 
	
	"""

	I = np.ones(len(r));

	r_i = r[0]; 
	r_o = r[-1];
	b   = -(r_i + r_o); 
	c   = r_i*r_o;
	
	S        = np.diag(1.0/((r**2)+b*r+c*I));
	S[0,0]   = 0.0; 
	S[-1,-1] = 0.0;

	D2 = D@D;
	D3 = D@D2;
	D4 = D2@D2;
	
	# Define \tilde{D^4} + Implement BC
	L4 = np.diag(r**2 + b*r + c*I)@D4 + 4.0*np.diag(2.0*r + b*I)@D3 + 12.0*D2;
	D4 = L4@S; # (d/dr)^4
	
	return D4[1:-1,1:-1];

def R2(R,N_fm):
	
	#print("Warning: make sure to call R2.dot(np vector) as spase matrix")

	from scipy.sparse  import diags, block_diag

	GR = diags( (R[1:-1]**2),0,format="csr"); 
	
	AT = [];
	for jj in range(N_fm): # [0,N_Fm -1] cosine basis
		if jj == 0:
			AT.append(1.*GR);
		else:		
			AT.append(GR);

	return block_diag(AT,format="csr");

def kGR_RT(R,N_fm,d): # Correct
	
	"""
	Compute the operator g(r) ∂_s f(r,s,t) in spectral space 

	as g(r) k f_k(r,t) = R_1/r^2 k  k f_k(r,t)

	returns the sparse matrix which performs this operation
	"""

	#print("Warning: make sure to call kGR_RT.dot(np vector) as spase matrix")

	nr  = len(R[1:-1]); 
	R_1 = 1./d;
	
	from scipy.sparse  import diags
	from scipy.sparse  import bmat

	GR = diags( (R_1**2)/(R[1:-1]**2), 0 ,format="csr")
	I  = diags( 0.*np.ones(nr)       , 0 ,format="csr")

	A1 = [];
	for jj in range(N_fm):
		AT = []; 
		j = jj + 1; # j Sine [1,N_Fm]
		for k in range(N_fm): # k Cosine [0,N_Fm-1] 
			if (k == j):
				AT.append(-k*GR);
			else:
				AT.append(I    );

		A1.append(AT)		

	return bmat(A1,format="csr")


@njit(fastmath=True) # Just check dT_0/dr form and sign
def DT0_theta(g, dT0,N_fm,nr, symmetric = False): 
	
	"""
	Implements the linear term

	(r^2)*J(ψ, T0) = (1/sinθ)*(∂(ψ sinθ)/∂θ)*T'0

	accounting for the interation of the base state T0 and ψ 
	multiplied by r^2 as done when solving for S or T.

	Here ψ is referenced as g.

	Inputs:

	g - numpy vector N_fm*nr = ψ
	dT0 -numpy vector nr
	N_fm - integer number of Fourier modes
	nr - number of Chebyshev points

	Returns:
	f -  numpy vector N_fm*nr = (r^2)*J(ψ, T0)

	"""

	f = np.zeros(g.shape); # In Cosine

	b = np.zeros(nr);
	for jj in range(0,N_fm,2): 
		
		j = (N_fm - 2 - jj ); # j cosine [0,2,4,....,N_Fm -2]
		ind_j = j*nr;

		#print("Row",j,"Cosine j=",j)
		
		if (j < (N_fm - 1 ) ):
			b += g[ind_j+nr:ind_j+2*nr];	

		if j == 0:
			f[ind_j:ind_j+nr] = dT0*1.0*b;
		else:	
			f[ind_j:ind_j+nr] = dT0*( (j + 1.0)*g[ind_j-nr:ind_j] + 2.0*b );	

	if symmetric == False:
		
		b = np.zeros(nr);
		for jj in range(1,N_fm,2): 
			
			j = (N_fm - jj); # j cosine [1,3,5,.....,N_Fm -1]
			ind_j = j*nr;
			
			#print("Row",j,"Cosine j=",j)

			if (j < (N_fm - 1 ) ):
				b += g[ind_j+nr:ind_j+2*nr];

			f[ind_j:ind_j+nr] = dT0*( (j + 1.0)*g[ind_j-nr:ind_j] + 2.0*b );

	return f;	

@njit(fastmath=True)
def A2_SINE(g,  D,R,N_fm,nr, symmetric = False): 

	"""
	Routine to perform operation

	f = A^2 ψ 

	imposing ψ = 0 at R_1,R_2

	Input:
	g - numpy vector N_fm*nr ~ ψ
	D - numpy matrix (nr+1,nr+1)
	R - numpy vector nr 
	N_fm - integer number of Fourier modes
	N_r - integer number of Chebyshev modes

	Returns:
	f - numpy vector N_fm*nr = A^2_{k,j} g_j 

	"""
	N = nr*N_fm; 
	f = np.zeros(N); 

	IR2 = 1.0/(R[1:-1]**2) 
	D2  = (D@D)[1:-1,1:-1]
	D2 = np.ascontiguousarray(D2);


	
	f_e = np.zeros(nr); 
	for jj in range(0,N_fm,2):

		j = N_fm-jj; 
		ind_j = (j-1)*nr; # Row ind

		#print("Evens Row row=%i"%(j-1),"Sine mode j=%i"%j)

		f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR2*( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
		f_e += g[ind_j:ind_j+nr];
	
	if symmetric == False:	
		f_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			j = N_fm-jj; 
			ind_j = (j-1)*nr; # Row ind

			#print("Odds Row row=%i"%(j-1),"Sine mode j=%i"%j)

			f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR2*( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
			f_e += g[ind_j:ind_j+nr];	

	return f;

# ~~~~~~~~ Time-stepping functions ~~~~~~~~~~~

# O(Nr^3 N_theta) complexity
#~~~~~~~~~~~~~
@njit(fastmath=True)
def NAB2_BSub_TSTEP(g, R2_Nab2,R2,I,N_fm,nr,dt, symmetric = False):
	
	"""
	Performs a back substitution to solve for 

	(r^2 - dt*r^2*∆T)f = g,	 

	imposing f = 0 at R_1,R_2 where ∆ is the spherical laplacian.
	
    Inputs:
	g - numpy vector N_fm*nr
	R2_Nab2 - matrix operator (nr,nr) Nabla2(D,R) without theta depedancy r^2*∆T = r^2 T'' +2r T'
	R2 - matrix operator (nr,nr) np.diag(R[1:-1]**2);
	I - matrix operator (nr,nr) np.eye(nr)
	N_fm - integer number of theta modes
	nr - iteger number of Chebyshev polynomials
	dt - float time-step

    Returns:
    f - numpy vector N_fm*nr
	"""
	
	# In eqn L*f = g, returns f
	# Should parrelelize very well on 2 cores as decoupled
	N   = nr*N_fm; 
	f   = np.zeros(N) 

	# ~~~~~~~~~~ Odds ~~~~~~~~~~
	if symmetric == False:
		b = np.zeros(nr);
		for jj in range(0,N_fm,2):		
			
			j = (N_fm - (jj + 1) );
			ind_j = j*nr;
			
			#print("Odds Row j=%i"%j,"Cosine j=%i"%j)
			bj = -j*(j + 1.0)
			A  = R2 - dt*(R2_Nab2 + bj*I)

			if j < (N_fm - 2 ):
				ßk = -2.*(j + 2.);
				#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
				ßk = -dt*ßk;
				b += ßk*f[(j+2)*nr:(j+3)*nr];

			#if j == 0:	
			#	f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - 0.5*b);	
			#else:
			f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - b);

	# ~~~~~~~~~ Evens ~~~~~~~~~~~~			
	b = np.zeros(nr);
	for jj in range(1,N_fm,2):		
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;

		#print("Evens Row j=%i"%j,"Cosine j=%i"%j)
		bj = -j*(j + 1.0)
		if j == 0:
			A = 1.*(R2 - dt*R2_Nab2)
		else:	
			A = R2 - dt*(R2_Nab2 + bj*I)
		
		if j < (N_fm - 2 ):
			ßk = -2.*(j + 2.);
			#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
			ßk = -dt*ßk;
			b += ßk*f[(j+2)*nr:(j+3)*nr];
		
		if j == 0:	
			f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - 0.5*b);	
		else:
			f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - b);

	return f;

@njit(fastmath=True)
def A4_BSub_TSTEP(g,  D4,IR4, D2,A2,IR2, N_fm,nr,dt, symmetric = False):

	"""
	Performs a back substitution to solve for 

	(A^2 - ∆t*Pr*A^2A^2)ψ = g

	imposing ψ = ψ' = 0 at boundaries 
	where ψ is denoted by f in this function.

	rather than adding the argument Pr we pass 

	dt = ∆t*Pr

	this premultiplying by Pr.

	Inputs:
	g - numpy vector N_fm*nr
	
	D4 - matrix operator (nr,nr) - Nabla4(D,R);
	IR4 - matrix operator (nr,nr) - diag(1/r^4)

	D2   - matrix operator (nr,nr) - np.matmul(IR2 ,2.0*D_sq - 4.0*(IR@D) + 6.0*IR2 )[1:-1,1:-1];
	A2   - matrix operator (nr,nr) - D@D
	IR2  - matrix operator (nr,nr) - diag(1/r^2)

	N_fm - integer number of theta modes
	nr - iteger number of Chebyshev polynomials
	dt - float time-step

	Returns:
	f - numpy vector N_fm*nr

	"""

	N = nr*N_fm; f = np.zeros(N); 
	
	
	# ~~~~~~~~~~~~~~~ EVENS ~~~~~~~~~~~~~~~~~~~~
	f_e = np.zeros(nr); bf_e = np.zeros(nr); 
	for jj in range(0,N_fm,2):

		row = (N_fm - (jj + 1) ); 
		ind_j = row*nr; # Row ind
		j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
		bj = -j*(j + 1.); bjt = -2.*j;
		
		#print("Evens Row row=%i"%row,"Sine mode j=%i"%j)

		L1 = D2 + bj*IR4; 

		L = (A2 + bj*IR2) - dt*(D4 + bj*L1);

		

		if row < (N_fm - 2 ):
		
			f_e += f[(row+2)*nr:(row+3)*nr];

			# Add time component
			b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
			
			f[ind_j:ind_j+nr] = np.linalg.solve(L,g[ind_j:ind_j+nr]+b_test);   #O(Nr^3 N_theta)


			# Add sums after to get +2 lag
			bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

		else:
			f[ind_j:ind_j+nr] = np.linalg.solve(L,g[ind_j:ind_j+nr]);
			bf_e += bj*f[ind_j:ind_j+nr];

	if symmetric == False:
		# ~~~~~~~~~~~~~~~ ODDS ~~~~~~~~~~~~~~~~~~~~		
		f_e = np.zeros(nr); bf_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			row = (N_fm - (jj + 1) ); 
			ind_j = row*nr; # Row ind
			j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
			bj = -j*(j + 1.); bjt = -2.*j;
			
			#print("Odds Row row=%i"%row,"Sine mode j=%i"%j)

			L1 = D2 + bj*IR4
			
			L = (A2 + bj*IR2) - dt*(D4 + bj*L1);

			if row < (N_fm - 2 ):
			
				f_e += f[(row+2)*nr:(row+3)*nr];

				# Add time component
				b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
				
				f[ind_j:ind_j+nr] = np.linalg.solve(L,g[ind_j:ind_j+nr]+b_test);   #O(Nr^3 N_theta)

				# Add sums after to get +2 lag
				bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

			else:

				f[ind_j:ind_j+nr] = np.linalg.solve(L,g[ind_j:ind_j+nr]);
				bf_e += bj*f[ind_j:ind_j+nr];

	return f;
#~~~~~~~~~~~~~

@njit(fastmath=True)
def J_theta_RT(g,     nr,N_fm, symmetric = False):
	
	f = np.zeros(g.shape); # In Cosine

	b = np.zeros(nr);
	for jj in range(0,N_fm,2): 
		
		j = (N_fm - 2 - jj ); # j cosine [0,2,4,....,N_Fm -2]
		ind_j = j*nr;

		#print("Row",j,"Cos(j*x) =",j)
		
		if (j < (N_fm - 1 ) ):
			b += g[ind_j+nr:ind_j+2*nr];	

		if j == 0:
			f[ind_j:ind_j+nr] = 1.0*b;
		else:	
			f[ind_j:ind_j+nr] = (j + 1.0)*g[ind_j-nr:ind_j] + 2.0*b;	

	if symmetric == False:
		
		b = np.zeros(nr);
		for jj in range(1,N_fm,2): 
			
			j = (N_fm - jj); # j cosine [1,3,5,.....,N_Fm -1]
			ind_j = j*nr;
			
			#print("Row",j,"Cos(j*x)=",j)

			if (j < (N_fm - 1 ) ):
				b += g[ind_j+nr:ind_j+2*nr];

			f[ind_j:ind_j+nr] = (j + 1.0)*g[ind_j-nr:ind_j] + 2.0*b;		 

	return f;

@njit(fastmath=True)
def A2_SINE_R2(g, N_fm,nr,D,R, symmetric = False): 

	"""
	Routine to mutiple psi ~ g by the matrix (1/r^2)*A^2_{k,j} in spectral space

	Input:
	g - numpy vector N_fm*nr ~ ψ
	D - numpy matrix (nr+1,nr+1)
	R - numpy vector nr 
	N_fm - integer number of Fourier modes
	N_r - integer number of Chebyshev modes

	Returns:
	f - numpy vector N_fm*nr = (1/r^2)*A^2_{k,j} g_j 

	"""

	N = nr*N_fm; 
	f = np.zeros(N); 

	IR4 = np.diag( 1.0/(R[1:-1]**4)); 
	IR4 = np.ascontiguousarray(IR4);
	D2  = ( np.diag( (1.0/R**2) )@(D@D) )[1:-1,1:-1]
	D2 = np.ascontiguousarray(D2);

	
	f_e = np.zeros(nr); 
	for jj in range(0,N_fm,2):

		j = N_fm-jj; # k_s wave-number, will be even
		ind_j = (j-1)*nr; # Row ind

		#print("Evens Row row=%i"%(j-1),"Sin(j*x)=%i"%j)

		f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR4.dot( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
		f_e += g[ind_j:ind_j+nr];

	
	if symmetric == False:	
		f_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			j = N_fm-jj; # k_s wave-number, will be odd
			ind_j = (j-1)*nr; # Row ind

			#print("Odds Row row=%i"%(j-1),"Sin(j*x)=%i"%j)

			f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR4.dot( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
			f_e += g[ind_j:ind_j+nr];	

	return f;

#~~~~~~~ Validated up to here ~~~~~~~~~~

@njit(fastmath=True)
def Vecs_To_NX(PSI,T,C, N_fm,nr, symmetric = False):

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	N  = N_fm*nr;
	NX = np.zeros(3*N);
	if symmetric == True:
		
		# O(N_fm/2) Correct

		for ii in range(1,N_fm,2): 
			
			#print("Row ii=%i, Sin(k_s*x) = %i"%(ii,ii+1))
			# a) psi parts
			ind_p = ii*nr;
			NX[ind_p:ind_p+nr] = PSI[:,ii];
		
		for ii in range(0,N_fm,2): 	
			
			#print("Row ii=%i, Cos(k_c*x) = %i"%(ii,ii))
			# b) T parts
			ind_T = N + ii*nr; 
			NX[ind_T:ind_T+nr] = T[:,ii];
			
			# c) C parts
			ind_C = 2*N + ii*nr;
			NX[ind_C:ind_C+nr] = C[:,ii];
	
	elif symmetric == False:	
		
		# O(N_fm) Correct

		for ii in range(N_fm): 
			
			# a) psi parts
			ind_p = ii*nr;
			NX[ind_p:ind_p+nr] = PSI[:,ii];
			
			# b) T parts
			ind_T = N + ii*nr; 
			NX[ind_T:ind_T+nr] = T[:,ii];
			
			# c) C parts
			ind_C = 2*N + ii*nr;
			NX[ind_C:ind_C+nr] = C[:,ii];

	return NX;	

@njit(fastmath=True)
def Derivatives(X_hat,JPSI,OMEGA, Dr, N_fm,nr, symmetric = False):

	sp = (nr, N_fm);#int( 3*(N_fm/2)) ); # ZERO-PADDED FOR DE-ALIASING !!!!!
	N  = N_fm*nr;

	# DCT's
	JT_psi_hat  = np.zeros(sp); 
	kDpsi_hat   = np.zeros(sp);
	komega_hat  = np.zeros(sp);         
	DT_hat 		= np.zeros(sp); 
	DC_hat 		= np.zeros(sp); 
	
	# DST's
	omega_hat = np.zeros(sp);  
	Dpsi_hat  = np.zeros(sp); 
	kT_hat 	  = np.zeros(sp); 
	kC_hat    = np.zeros(sp);
	
	# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm 

	if symmetric == True: 

		# O(nr^2*N_fm/2)
		for ii in range(1,N_fm,2): # Sine [1,N_fm]

			k_s = ii + 1; # [1,N_fm]
			
			#print("Row ii=%i, Sin(k_s*x) = %i"%(ii,k_s))

			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~  # Correct
			ind_p = ii*nr; 
			psi   = X_hat[ind_p:ind_p+nr];

			Dpsi_hat[:,ii]    = Dr.dot(psi);
			kDpsi_hat[:,ii]   = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine
					
		
			omega_hat[:,ii]   = OMEGA[ind_p:ind_p+nr];
			komega_hat[:,ii]  = k_s*omega_hat[:,ii] # Sine -> Cosine 

		for ii in range(0,N_fm,2): # cosine [0,N_fm-1]
			
			k_c = ii;     # [0,N_fm-1]

			#print("Row ii=%i, Cosine(k_c*x) = %i"%(ii,k_c) )

			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~  # Correct
			ind_p = ii*nr; 
			JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr]; # Sine -> Cosine

			# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
			ind_T = N + ii*nr; 
			T 	  = X_hat[ind_T:ind_T+nr];

			DT_hat[:,ii] = Dr.dot(T);
			kT_hat[:,ii] = -k_c*T; # Cosine -> Sine

			# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
			ind_C = 2*N + ii*nr; 
			C 	  = X_hat[ind_C:ind_C+nr];

			DC_hat[:,ii] = Dr.dot(C);
			kC_hat[:,ii] = -k_c*C; # Cosine -> Sine

	elif symmetric == False:
		
		# O(nr^2*N_fm)
		for ii in range(N_fm):

			# Wavenumbers
			k_s = ii + 1; # [1,N_fm  ]
			k_c = ii;     # [0,N_fm-1]
			
			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ ???
			ind_p = ii*nr; 
			psi   = X_hat[ind_p:ind_p+nr];

			Dpsi_hat[:,ii]    = Dr.dot(psi);
			kDpsi_hat[:,ii]   = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine #
					
			JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr];

			omega_hat[:,ii]   = OMEGA[ind_p:ind_p+nr];
			komega_hat[:,ii]  = k_s*omega_hat[:,ii] # Sine -> Cosine 


			# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
			ind_T = N + ii*nr; 
			T 	  = X_hat[ind_T:ind_T+nr];

			DT_hat[:,ii] = Dr.dot(T);
			kT_hat[:,ii] = -k_c*T; # Cosine -> Sine

			# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
			ind_C = 2*N + ii*nr; 
			C 	  = X_hat[ind_C:ind_C+nr];

			DC_hat[:,ii] = Dr.dot(C);
			kC_hat[:,ii] = -k_c*C; # Cosine -> Sine

	
	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;

	# b) sine -> cosine
	kDpsi_hat[:,1:]  = kDpsi_hat[:,0:-1];  kDpsi_hat[:,0]  = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;

	return JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat;

def NLIN_FX(X_hat,	inv_D,D,R,N_fm,nr, symmetric = False, kinetic = False):

	"""

	Compute the nonlinear terms by taking the: 

	∂_s X(r,s) -> -k_s*X or -k_c*X, polar derivatives

	∂_r X(r,s) -> D*X, radial derivatives 

	return F(X,X) a vetor same shape as X

	"""

	N  = nr*N_fm; 
	if N_fm%2 != 0:
		print("\n\n Must choose an even number of Fourier modes N_fm = 2*N_fm !!! \n\n");
		sys.exit();

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	JPSI  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric) # ~ cos(k_c*x)
	OMEGA = A2_SINE_R2(X_hat[0:N], N_fm,nr,D,R, symmetric); # ~ sin(k_s*x)
	Dr    = D[1:-1,1:-1];
	JPSI  = np.ascontiguousarray(JPSI);
	OMEGA = np.ascontiguousarray(OMEGA);
	Dr    = np.ascontiguousarray(Dr);
	
	# 1) Compute derivatives & Transform to Nr x N_fm
	JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat = Derivatives(X_hat,JPSI,OMEGA, Dr,N_fm,nr, symmetric);
	
	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# psi, T,C 
	JT_psi = IDCT(JT_psi_hat,n = (3*N_fm)//2) # Projected okay
	komega = IDCT(komega_hat,n = (3*N_fm)//2) # needs shift
	kDpsi  = IDCT( kDpsi_hat,n = (3*N_fm)//2) # needs shift
	DT 	   = IDCT(    DT_hat,n = (3*N_fm)//2) # Projected okay 
	DC     = IDCT(    DC_hat,n = (3*N_fm)//2) # Projected okay

	# psi, T, C
	omega  = IDST( omega_hat,n = (3*N_fm)//2) # Projected okay
	Dpsi   = IDST(  Dpsi_hat,n = (3*N_fm)//2) # Projected okay
	kT 	   = IDST(    kT_hat,n = (3*N_fm)//2) # Needs shift
	kC 	   = IDST(    kC_hat,n = (3*N_fm)//2) # Needs shift

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*omega) - (kDpsi*omega + Dpsi*komega);
	NJ_PSI_T = JT_psi*DT - Dpsi*kT;	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;

	# 4) Compute DCT and DST & un-pad
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = DST(NJ_PSI__,n=N_fm);	
	J_PSI_T_hat = DCT(NJ_PSI_T,n=N_fm);	
	J_PSI_C_hat = DCT(NJ_PSI_C,n=N_fm);		

	if kinetic == True:

		KE = Kinetic_Energy(JT_psi,Dpsi, R,inv_D,N_fm,nr);
		return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	N_fm,nr, symmetric), KE;
	else:
		return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	N_fm,nr, symmetric);	

def Kinetic_Energy(Jψ,dr_ψ, R,inv_D,N_fm,nr):

	"""

	Compute the volume integrated kinetic energy

	KE = (1/2)*(1/V) int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ

	where

	V = int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ = (2/3)*(r2^3 - r1^3)
	
	"""

	IR2  = np.diag(1./(R[1:-1]**2));
	IR2  = np.ascontiguousarray(IR2);

	KE_rθ = IR2@(Jψ**2) + abs(dr_ψ)**2;
	
	# Integrate in θ and take zero mode essentially the IP with 
	KE_r       = 0.*R;
	#from scipy.fft import dst
	#KE_r[1:-1] = dst(KE_rθ,type=2,axis=-1,overwrite_x=True)[:,0];
	#KE_r[1:-1]/= N_fm
	from Transforms import DST
	KE_r[1:-1] = DST(KE_rθ,n=N_fm)[:,0]

	# Multiply by r^2 and integrate w.r.t r
	#KE_r = R*R*KE_r;
	
	KE = inv_D[0,:].dot(KE_r[0:-1]);
	V  = (2./3.)*abs(R[-1]**3 - R[0]**3);

	return (.5/V)*KE;

def NLIN_DFX(dv_hat,X_hat,	inv_D,D,R,N_fm,nr, symmetric = False):

	"""

	Compute the Jacobian of the nonlinear terms F(X) by taking the: 

	∂_s X(r,s) -> -k_s*X or -k_c*X, polar derivatives

	∂_r X(r,s) -> D*X, radial derivatives 

	return DF(X)*dv = N(X,dv) + N(dv,X) a vetor same shape as X

	"""

	from scipy.fft import dct, idct, dst, idst

	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	N  = nr*N_fm; 
	if N_fm%2 != 0:
		print("\n\n Must choose an even number of Fourier modes N_fm = 2*N_fm !!! \n\n");
		sys.exit();

	# A)  Base state X terms
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	JPSI  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric)
	OMEGA = A2_SINE_R2(X_hat[0:N], N_fm,nr,D,R, symmetric);
	Dr    = D[1:-1,1:-1];
	JPSI  = np.ascontiguousarray(JPSI);
	OMEGA = np.ascontiguousarray(OMEGA);
	Dr    = np.ascontiguousarray(Dr);
	
	# A.1) Compute derivatives & Transform to Nr x N_fm
	JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat = Derivatives(X_hat,JPSI,OMEGA, Dr,N_fm,nr, symmetric);
	
	# A.2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# psi, T,C 
	JT_psi = idct(JT_psi_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	komega = idct(komega_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # needs shift
	kDpsi  = idct( kDpsi_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # needs shift
	DT 	   = idct(    DT_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay 
	DC     = idct(    DC_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay

	# psi, T, C
	omega  = idst( omega_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	Dpsi   = idst(  Dpsi_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	kT 	   = idst(    kT_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Needs shift
	kC 	   = idst(    kC_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Needs shift


	# B)  Perturbation ∆X terms
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	δJψ = J_theta_RT(dv_hat[0:N], nr,N_fm, symmetric)
	δΩ  = A2_SINE_R2(dv_hat[0:N], N_fm,nr,D,R, symmetric);
	#Dr    = D[1:-1,1:-1];
	δJψ = np.ascontiguousarray(δJψ);
	δΩ  = np.ascontiguousarray(δΩ);
	#Dr    = np.ascontiguousarray(Dr);
	
	# ) Compute derivatives & Transform to Nr x N_fm
	δJT_ψ_hat,δkDψ_hat,δkΩ_hat,δDT_hat,δDC_hat,δΩ_hat,δDψ_hat,δkT_hat,δkC_hat = Derivatives(dv_hat,δJψ,δΩ, Dr,N_fm,nr, symmetric);
	
	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# psi, T,C 
	δJT_ψ = idct(δJT_ψ_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	δkΩ   = idct(  δkΩ_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # needs shift
	δkDψ  = idct( δkDψ_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # needs shift
	δDT   = idct(  δDT_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay 
	δDC   = idct(  δDC_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay

	# psi, T, C
	δΩ    = idst(   δΩ_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	δDψ   = idst(  δDψ_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Projected okay
	δkT   = idst(  δkT_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Needs shift
	δkC   = idst(  δkC_hat,type=2,axis=-1,overwrite_x=True,n = (3*N_fm)//2) # Needs shift


	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*δΩ)   - (kDpsi*δΩ   + Dpsi*δkΩ);
	NJ_PSI__+= Dr@(δJT_ψ*omega) - (δkDψ*omega + δDψ*komega)
	NJ_PSI_T = (δJT_ψ*DT - δDψ*kT) + (JT_psi*δDT - Dpsi*δkT);	
	NJ_PSI_C = (δJT_ψ*DC - δDψ*kC) + (JT_psi*δDC - Dpsi*δkC);

	# 4) Compute DCT and DST, un-pad, multiply by scaling factor 1/N_fm
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = dst(NJ_PSI__,type=2,axis=-1,overwrite_x=True)[:,0:N_fm];
	J_PSI_T_hat = dct(NJ_PSI_T,type=2,axis=-1,overwrite_x=True)[:,0:N_fm]; 
	J_PSI_C_hat = dct(NJ_PSI_C,type=2,axis=-1,overwrite_x=True)[:,0:N_fm];		

	
	return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	N_fm,nr, symmetric);	


# ~~~~~~~~~~~~ Interpolation functions ~~~~~~~~~~~~~~

def INTERP_RADIAL(N_n,N_o,X_o,d):

	if N_n == N_o:
		return X_o;

	_,R_n=cheb_radial(N_n,d)
	nr_n = len(R_n[1:-1]);

	_,R_o=cheb_radial(N_o,d)
	nr_o = len(R_o[1:-1]); 
	
	N_fm = len(X_o)//(3*nr_o);
	X_n  = np.zeros(3*nr_n*N_fm);
	
	for k in range(N_fm):

		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		ind_o = nr_o*k;	
		PSI   = np.polyfit(R_o,np.hstack( ([0.], X_o[ind_o:ind_o+nr_o] ,[0.])   ),len(R_o));	
			
		ind_n = nr_n*k;
		# Polyvals to collocation space on new grid
		X_n[ind_n:ind_n+nr_n] = np.polyval(PSI,R_n[1:-1])
		
		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		ind_o = N_fm*nr_o + nr_o*k;		
		T     = np.polyfit(R_o,np.hstack( ([0.], X_o[ind_o:ind_o+nr_o] ,[0.])   ),len(R_o));	

		ind_n = N_fm*nr_n + nr_n*k;
		X_n[ind_n:ind_n+nr_n] = np.polyval(T,R_n[1:-1])
		
		# ~~~~ S ~~~~~~~~~~~~~~~~~~
		ind_o = 2*N_fm*nr_o + nr_o*k;
		S = np.polyfit(R_o,np.hstack( ([0.], X_o[ind_o:ind_o+nr_o] ,[0.])   ),len(R_o));	
		
		ind_n = 2*N_fm*nr_n + nr_n*k;
		X_n[ind_n:ind_n+nr_n] = np.polyval(S,R_n[1:-1])
	
	return X_n;

def INTERP_THETAS(N_fm_n,N_fm_o,X_o):

	if N_fm_n == N_fm_o:
		return X_o;

	from scipy.fft import dct, idct, dst, idst

	nr  = len(X_o)//(3*N_fm_o);
	XX = np.zeros(3*nr*N_fm_n);
	
	if N_fm_o < N_fm_n:
	
		PSI_X = np.zeros((nr,N_fm_n)); 
		T_X   = np.zeros((nr,N_fm_n)); 
		S_X   = np.zeros((nr,N_fm_n)); 
	
	elif N_fm_o > N_fm_n:
	
		PSI_X = np.zeros((nr,N_fm_o)); 
		T_X   = np.zeros((nr,N_fm_o)); 
		S_X   = np.zeros((nr,N_fm_o));


	# 1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for k in range(N_fm_o):
		
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		ind = k*nr;
		PSI_X[:,k] = X_o[ind:ind+nr] 
		
		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		ind = N_fm_o*nr + k*nr;
		T_X[:,k] = X_o[ind:ind+nr] 

		# ~~~~ S ~~~~~~~~~~~~~~~~~~
		ind = 2*N_fm_o*nr + k*nr;
		S_X[:,k] = X_o[ind:ind+nr] 


	# 2) iDCT or iDST Interpolate onto a grid 
	PSI_X = idst(PSI_X,type=2,axis=-1,overwrite_x=True) 
	T_X   = idct(T_X  ,type=2,axis=-1,overwrite_x=True) 
	S_X   = idct(S_X  ,type=2,axis=-1,overwrite_x=True) 


	# 3) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	PSI_hat = dst(PSI_X,type=2,axis=-1,overwrite_x=True)[:,0:N_fm_n]
	T_hat   = dct(T_X  ,type=2,axis=-1,overwrite_x=True)[:,0:N_fm_n]
	S_hat   = dct(S_X  ,type=2,axis=-1,overwrite_x=True)[:,0:N_fm_n]

	# 4) DCT or DST onto more polynomials
	for k in range(N_fm_n):
		
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		ind = k*nr
		XX[ind:ind +nr] = PSI_hat[:,k];

		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		ind = N_fm_n*nr + k*nr
		XX[ind:ind+nr] = T_hat[:,k];

		# ~~~~ S ~~~~~~~~~~~~~~~~~~
		ind = 2*N_fm_n*nr + k*nr
		XX[ind:ind+nr] = S_hat[:,k];

	return XX;	

# ~~~~~ Full Nr x N_fm blocks ~~~~

# ~~~~~ NABLA2 Cosine~~~~ Correct as of 16/09/22
def A2_theta_C(R,N_fm): # No 1/R^2
	
	# LAP2_theta Cosine-Basis  = r^2 \nabla^2 + A^2_\theta

	from scipy.sparse  import bmat
	
	nr = len(R[1:-1]); 
	IR = np.eye(nr); #diags( np.ones(nr),0,format="csr");  #

	AT = [];
	for j in range(N_fm): # [0,N_Fm -1]
		AT_j = [];
		for k in range(N_fm): # [0,N_Fm -1]
			
			if (k == j):
				AT_j.append(-k*(k + 1.0)*IR)
			elif (k > j) and ( (k+j)%2 == 0 ):
				AT_j.append(-k*2.0*IR);		
			else:
				AT_j.append(None);
		AT.append(AT_j);				 
	
	return bmat(AT,format="csr");

def NABLA2_COS(D,R,N_fm):
	
	from scipy.sparse  import block_diag

	DR = Nabla2(D,R); # r^2 T'' +2r T'

	LAP_2 = [];
	for k in range(N_fm): # [0,N_Fm -1] cosine basis	
		if k == 0:
			LAP_2.append(2.*DR);
		else:	
			LAP_2.append(DR);

	return block_diag(LAP_2,format="csr") + A2_theta_C(R,N_fm);

# ~~~~~ J_theta Cosine-Basis~~~~ Correct as of 16/09/22
def T0J_theta(R,N_fm,d): 
	
	# Includes -T'_0; #/r^2
	from scipy.sparse  import diags
	from scipy.sparse  import bmat

	R_1 = 1./d; 
	R_2 = (1. + d)/d;
	A_t = (R_1*R_2)/(R_1-R_2)

	IR = A_t*diags( 1.0/(R[1:-1]**2),0,format="csr"); 
	nr = len(R[1:-1]); #print("nr",nr);
	#IR = np.eye(nr);

	JT = [];
	for j in range(N_fm): # j cosine [0,N_Fm -1]
		AT_j = [];
		for kk in range(N_fm): 
			k = kk + 1; # k Sine [1,N_Fm]
			
			if (k == j) and (j > 0):
				AT_j.append( (k + 1.0)*IR); 	
			elif (k > j) and ( (j+k)%2 == 0 ):	
				AT_j.append(2.*IR);
			else:
				if j == 0:
					AT_j.append( np.zeros((nr,nr)) );
				else:
					AT_j.append( None );	

		JT.append(AT_j);				 
			
	return bmat(JT,format="csr");

def J_theta(R,N_fm): 
	
	# Includes -T'_0; #/r^2
	from scipy.sparse  import bmat
 
	nr = len(R[1:-1]); #print("nr",nr);
	IR = np.eye(nr);

	JT = [];
	for j in range(N_fm): # j cosine [0,N_Fm -1]
		AT_j = [];
		for kk in range(N_fm): 
			k = kk + 1; # k Sine [1,N_Fm]
			
			if (k == j) and (j > 0):
				AT_j.append( (k + 1.0)*IR); 	
			elif (k > j) and ( (j+k)%2 == 0 ):
				AT_j.append(2.*IR);
			else:
				if j == 0:
					AT_j.append( np.zeros((nr,nr)) );
				else:
					AT_j.append( None );	

		JT.append(AT_j);				 
			
	return bmat(JT,format="csr");


# ~~~~~ g(r)_d_theta ~~~~ not rechecked
def kGR(R,N_fm,d): # Correct
	
	from scipy.sparse import diags,block_diag,identity
	nr = len(R[1:-1]); R_1 = 1./d;
	
	GR = diags( (R_1**2)/(R[1:-1]**2), 0 ,format="csr")

	# Alternatively goes like GR = I ?
	
	AT = [];
	for jj in range(N_fm): 
		j = jj + 1; # j Sine [1,N_Fm]
		for k in range(N_fm): # k Cosine [0,N_Fm-1] 
			if (k == j):

				AT.append(-k*GR);
	
	A1 = block_diag(AT,format="csr");

	# AT must be an 
	#ID = diags(AT, 1,format="csr")

	ID = 0.*identity(nr*N_fm,format="csr");
	ID[:-1*nr,1*nr:] = A1; # FIX V slow here

	return ID;


# ~~~~~ NABLA4 Sine~~~~ Correct as of 19/09/22
def A2_theta_S(R,N_fm): 
	
	# LAP2_theta Sine-Basis, D^2 + (A^2_\theta)/r^2

	from scipy.sparse import diags,bmat
	
	IR = diags( 1.0/(R[1:-1]**2),0,format="csr"); #np.ones(len(R[1:-1]));# 
	
	AT = [];
	for jj in range(N_fm): # Sine [1,N_Fm] 
		j = jj + 1; 
		
		AT_j = [];
		for kk in range(N_fm): # Sine [1,N_Fm] 
			k = kk + 1; 

			if (k == j):
				AT_j.append(-j*(j + 1.0)*IR);
			elif (k > j) and ( (k + j)%2 == 0 ):
				AT_j.append(-2.0*j*IR);
			else:
				AT_j.append(None);
		AT.append(AT_j);				 
	
	return bmat(AT,format="csr");

def NABLA2_SINE(D,R,N_fm):

	from scipy.sparse import block_diag

	DR = (D@D)[1:-1,1:-1];

	LAP_2 = []; 
	for j in range(N_fm): # [1,N_Fm -1] sine basis
		LAP_2.append(DR);
	
	D2 = block_diag(LAP_2,format="csr");
	
	return D2 + A2_theta_S(R,N_fm);

def NABLA4_SINE(D,R,N_fm):

	from scipy.sparse import block_diag
	# LAP4 full Sine-Basis, includes 1/r^2

	nr  = len(R[1:-1]); 
	IR2 = np.diag( 1.0/(R**2) ); 
	IR  = np.diag(1.0/R); # Keep dense
	
	LAP_4 = []; DT = []; 
	D4 = Nabla4(D,R); 
	D2 = 2.0*(D@D) - 4.0*(IR@D) + 6.0*IR2; 
	for k in range(N_fm): # Sine [1,N_Fm]
		LAP_4.append(D4);
		DT.append(D2[1:-1,1:-1]);

	LAP4 	 = block_diag(LAP_4,format="csr").toarray()
	DTT 	 = block_diag(DT,format="csr").toarray()
	A2_theta = A2_theta_S(R,N_fm).toarray()

	return LAP4 + A2_theta@DTT + A2_theta@A2_theta; 

#D,R=cheb_radial(4,1)
#print(A2_theta_S(R,4).toarray())



# O(Nr^2 N_theta) complexity & Memory
#~~~~~~~~~~~~~
def NAB2_TSTEP_MATS(dt,N_fm,nr,D,R):


	N = nr*N_fm; 
	M0 = np.zeros((nr,nr,N_fm));
	
	N2R = np.diag(R[1:-1]**2) -dt*Nabla2(D,R); 
	I   = -dt*np.eye(nr);

	for jj in range(N_fm):		
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;
		
		bj = -float(j)*( float(j) + 1.0)

		if j == 0:
			M0[:,:,jj] = np.linalg.inv( 2.0*N2R  );	
		else:
			M0[:,:,jj] = np.linalg.inv( N2R + bj*I  );
	
	return M0;

@njit(fastmath=True)
def NAB2_BSub_TSTEP_V2(g, L_inv,N_fm,nr,dt, symmetric = False):
	
	"""
	Performs a back substitution to solve for 

	(r^2 - dt*r^2*∆T)f = g,	 

	where ∆ is the spherical laplacian.
	
	Inputs:

	L_inv = (r^2 - dt*r^2*∆T)^-1
	"""
	
	# In eqn L*f = g, returns f
	# Should parrelelize very well on 2 cores as decoupled
	N   = nr*N_fm; 
	f   = np.zeros(N) 

	# ~~~~~~~~~~ Odds ~~~~~~~~~~
	if symmetric == False:
		b = np.zeros(nr);
		for jj in range(0,N_fm,2):		
			
			j = (N_fm - (jj + 1) );
			ind_j = j*nr;
			A  = np.ascontiguousarray(L_inv[:,:,jj]);

			#print("Odds Row j=%i"%j)
			bj = -j*(j + 1.0)

			if j < (N_fm - 2 ):
				ßk = -2.*(j + 2.);
				#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
				ßk = -dt*ßk;
				b += ßk*f[(j+2)*nr:(j+3)*nr];

			#if j == 0:	
			#	f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr] - .5*b);	
			#else:
			f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr] - b);

	# ~~~~~~~~~ Evens ~~~~~~~~~~~~			
	b = np.zeros(nr);
	for jj in range(1,N_fm,2):		
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;
		A  = np.ascontiguousarray(L_inv[:,:,jj]);

		#print("Evens Row j=%i"%j)
		bj = -j*(j + 1.0)
		
		if j < (N_fm - 2 ):
			ßk = -2.*(j + 2.);
			#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
			ßk = -dt*ßk;
			b += ßk*f[(j+2)*nr:(j+3)*nr];
		
		#if j == 0:	
		#	f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr] - .5*b);	
		#else:
		f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr] - b);	

	return f;

# ERROR IN THESE TWO

def A4_TSTEP_MATS(  dt,N_fm,nr,D,R):
	
	M0 = np.zeros((nr,nr,N_fm));

	IR2 = np.diag( 1.0/(R**2) ); 
	IR  = np.diag(1.0/R); 
	IR4 = np.diag( 1.0/(R[1:-1]**4) );
	D_sq= np.matmul(D,D); 
	D4  = Nabla4(D,R); 
	D2  = np.matmul( IR2 , 2.0*D_sq - 4.0*np.matmul(IR,D) + 6.0*IR2 )[1:-1,1:-1]; 
	A2  = D_sq[1:-1,1:-1]; 
	IR2 = IR2[1:-1,1:-1];

	for jj in range(N_fm):

		row = (N_fm - (jj + 1) ); 
		ind_j = row*nr; # Row ind
		j = float(N_fm-jj); # Remeber sine counting is from 1 - N_theta
		bj = -j*(j + 1.); bjt = -2.*j;
		
		L1 = D2 + bj*IR4; 

		L = (A2 + bj*IR2) - dt*(D4 + bj*L1);

		M0[:,:,jj] = np.linalg.inv(L);

	return M0;

@njit(fastmath=True)
def A4_BSub_TSTEP_V2(g,  L_inv,D,R,N_fm,nr,dt, symmetric = False):

	"""
	Performs a back substitution to solve for 

	(A^2 - ∆t*Pr*A^2A^2)ψ = g

	where ψ is denoted by f in this function.

	rather than adding the argument Pr we pass 

	dt = ∆t*Pr

	this premultiplying by Pr.


	Inputs:

	L_inv = (A^2 - ∆t*Pr*A^2A^2)^-1
	"""

	N = nr*N_fm; f = np.zeros(N); 
	
	IR  = np.diag(1.0/R); 
	IR  = np.ascontiguousarray(IR);
	IR2	= IR@IR;
	D   = np.ascontiguousarray(D);
	D2  = ( IR2@(2.0*(D@D) - 4.0*IR@D + 6.0*IR2 ) )[1:-1,1:-1];
	IR2 = IR2[1:-1,1:-1];
	IR2 = np.ascontiguousarray(IR2);
	IR4 = IR2@IR2;

	if symmetric == False:
		# ~~~~~~~~~~~~~~~ EVENS ~~~~~~~~~~~~~~~~~~~~
		f_e = np.zeros(nr); bf_e = np.zeros(nr); 
		for jj in range(0,N_fm,2):

			row = (N_fm - (jj + 1) ); 
			ind_j = row*nr; # Row ind
			j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
			bj = -j*(j + 1.); bjt = -2.*j;
			
			L1 = D2 + bj*IR4;
			A  = np.ascontiguousarray(L_inv[:,:,jj]);
			
			#print("Evens Row row=%i"%row)
			#print("Sine mode j=%i"%j)

			if row < (N_fm - 2 ):
			
				f_e += f[(row+2)*nr:(row+3)*nr];

				# Add time component
				b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
				f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr]+b_test);   #O(Nr^2 N_theta)


				# Add sums after to get +2 lag
				bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

			else:
				f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr]);
				bf_e += bj*f[ind_j:ind_j+nr];


	# ~~~~~~~~~~~~~~~ ODDS ~~~~~~~~~~~~~~~~~~~~		
	f_e = np.zeros(nr); bf_e = np.zeros(nr); 
	for jj in range(1,N_fm,2):

		row = (N_fm - (jj + 1) ); 
		ind_j = row*nr; # Row ind
		j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
		bj = -j*(j + 1.); bjt = -2.*j;
		
		#print("Odds Row row=%i"%row)
		#print("Sine mode j=%i"%j)

		L1 = D2 + bj*IR4
		A  = np.ascontiguousarray(L_inv[:,:,jj]);

		if row < (N_fm - 2 ):
		
			f_e += f[(row+2)*nr:(row+3)*nr];

			# Add time component
			b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
			f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr]+b_test);   #O(Nr^2 N_theta)

			# Add sums after to get +2 lag
			bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

		else:

			#f[ind_j:ind_j+nr] = np.linalg.solve(L,g[ind_j:ind_j+nr]);
			f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr]);
			bf_e += bj*f[ind_j:ind_j+nr];

	return f;

#~~~~~~~~~~~~~