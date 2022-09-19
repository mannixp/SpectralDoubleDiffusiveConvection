
#!/usr/bin/env python

from numba import njit
import numpy as np
import sys, os, time

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

'''
# Check the Matrix works
D,R=cheb_radial(10,1)
A  = Nabla2(D,R)

f = np.sin(R);
f_num = A.dot(f)
print(f_num);

f_anal = -np.sin(R)*(R**2) + 2*R*np.cos(R);
print(f_anal);

print(np.linalg.norm(f_anal - f_num,np.inf));
sys.exit()
'''

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

def Nabla4(D,r): # Didn't check
	
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

# Fixed an error corresponding to the zero mode needing to be multiplied by 2
def R2(R,N_fm):
	
	print("Warning: make sure to call R2.dot(np vector) as spase matrix")

	from scipy.sparse  import diags, block_diag

	GR = diags( (R[1:-1]**2),0,format="csr"); 
	
	AT = [];
	for jj in range(N_fm): # [0,N_Fm -1] cosine basis
		
		if jj == 0:
			AT.append(2.*GR);
		else:
			AT.append(GR);

	return block_diag(AT,format="csr");

def kGR_RT(R,N_fm,d): # Correct
	
	"""
	Compute the operator g(r) ∂_s f(r,s,t) in spectral space 

	as g(r) k f_k(r,t) = R_1/r^2 k  k f_k(r,t)

	returns the sparse matrix which performs this operation
	"""

	print("Warning: make sure to call kGR_RT.dot(np vector) as spase matrix")

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

	if symmetric == False:
		b = np.zeros(nr);
		for jj in range(0,N_fm,2): # j cosine [0,N_Fm -1]
			
			j = (N_fm - (jj + 1) );
			ind_j = j*nr;
			ind_k = (j -1)*nr;

			#print("j=",j)
			
			if (j < (N_fm - 1 ) ):
				b += g[ind_k+2*nr:ind_k+3*nr];	

			if j == 0:
				f[ind_j:ind_j+nr] = dT0*2.0*b;
			else:
				f[ind_j:ind_j+nr] = dT0*( (j + 1.0)*g[ind_k:ind_k+nr] + 2.0*b );	
			
			 
	b = np.zeros(nr);
	for jj in range(1,N_fm,2): # j cosine [0,N_Fm -1]
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;
		ind_k = (j -1)*nr;
		
		#print("j=",j)

		if (j < (N_fm - 1 ) ):
			b += g[ind_k+2*nr:ind_k+3*nr];

		if j == 0:
			f[ind_j:ind_j+nr] = dT0*2.0*b;
		else:	
			f[ind_j:ind_j+nr] = dT0*( (j + 1.0)*g[ind_k:ind_k+nr] + 2.0*b );		 

	return f;	

@njit(fastmath=True)
def A2_SINE(g,  D,R,N_fm,nr, symmetric = False): 

	"""
	Routine to perform operation

	f = A^2 ψ 

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

		f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR2*( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
		f_e += g[ind_j:ind_j+nr];
	
	if symmetric == False:
		
		f_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			j = N_fm-jj; 
			ind_j = (j-1)*nr; # Row ind

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

	where ∆ is the spherical laplacian.
	
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
			
			#print("Odds Row j=%i"%j)
			bj = -j*(j + 1.0)
			A  = R2 - dt*(R2_Nab2 + bj*I)

			if j < (N_fm - 2 ):
				ßk = -2.*(j + 2.);
				#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
				ßk = -dt*ßk;
				b += ßk*f[(j+2)*nr:(j+3)*nr];

			f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - b);

	# ~~~~~~~~~ Evens ~~~~~~~~~~~~			
	b = np.zeros(nr);
	for jj in range(1,N_fm,2):		
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;

		#print("Evens Row j=%i"%j)
		bj = -j*(j + 1.0)
		if j == 0:
			A = 2.*(R2 - dt*R2_Nab2)
		else:	
			A = R2 - dt*(R2_Nab2 + bj*I)
		
		if j < (N_fm - 2 ):
			ßk = -2.*(j + 2.);
			#print("ßk =%e, by k=%i:%i"%(ßk,j + 2,j + 3) )
			ßk = -dt*ßk;
			b += ßk*f[(j+2)*nr:(j+3)*nr];
			
		f[ind_j:ind_j+nr] = np.linalg.solve(A,g[ind_j:ind_j+nr] - b);	
		
	return f;


@njit(fastmath=True)
def A4_BSub_TSTEP(g,  D4,IR4, D2,A2,IR2, N_fm,nr,dt, symmetric = False):

	"""
	Performs a back substitution to solve for 

	(A^2 - ∆t*Pr*A^2A^2)ψ = g

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
			M0[:,:,jj] = np.linalg.inv( 2.*N2R  );	
		else:
			M0[:,:,jj] = np.linalg.inv( N2R + bj*I  );
	
	return M0;

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
			
		f[ind_j:ind_j+nr] = A.dot(g[ind_j:ind_j+nr] - b);	

	return f;

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

	# ~~~~~~~~~~~~~~~ EVENS ~~~~~~~~~~~~~~~~~~~~
	f_e = np.zeros(nr); bf_e = np.zeros(nr); 
	for jj in range(0,N_fm,2):

		row = (N_fm - (jj + 1) ); 
		ind_j = row*nr; # Row ind
		j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
		bj = -j*(j + 1.); bjt = -2.*j;
		
		L1 = D2 + bj*IR4;
		A  = np.ascontiguousarray(L_inv[:,:,jj]);
		

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


	if symmetric == False:		
		# ~~~~~~~~~~~~~~~ ODDS ~~~~~~~~~~~~~~~~~~~~		
		f_e = np.zeros(nr); bf_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			row = (N_fm - (jj + 1) ); 
			ind_j = row*nr; # Row ind
			j = N_fm-jj; # Remeber sine counting is from 1 - N_theta
			bj = -j*(j + 1.); bjt = -2.*j;
			
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


@njit(fastmath=True)
def J_theta_RT(g,     nr,N_fm, symmetric = False):
	
	f = np.zeros(g.shape); # In Cosine
	#g # in sine

	if symmetric == False:
		b = np.zeros(nr);
		for jj in range(0,N_fm,2): # j cosine [0,N_Fm -1], odds as j cosine [1,3,5,7,...,N_Fm -1]
			
			j = (N_fm - (jj + 1) );
			ind_j = j*nr;
			ind_k = (j -1)*nr; # k sine wave numbers [1,3,5,7,...,N_Fm -1] indecies k = [0,2,4,6,...,N_Fm]
			
			if (j < (N_fm - 1 ) ):
				b += g[ind_k+2*nr:ind_k+3*nr];	

			if j == 0:
				f[ind_j:ind_j+nr] = 2.0*b;
			else:	
				f[ind_j:ind_j+nr] = (j + 1.0)*g[ind_k:ind_k+nr] + 2.0*b;	


	b = np.zeros(nr);
	for jj in range(1,N_fm,2): # j cosine [0,N_Fm -1], evens as j cosine [0,2,4,5,...,N_Fm -1]
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;
		ind_k = (j -1)*nr;
		
		if (j < (N_fm - 1 ) ):
			b += g[ind_k+2*nr:ind_k+3*nr];

		if j == 0:
			f[ind_j:ind_j+nr] = 2.0*b;
		else:	
			f[ind_j:ind_j+nr] = (j + 1.0)*g[ind_k:ind_k+nr] + 2.0*b;		 

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

		f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR4.dot( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
		f_e += g[ind_j:ind_j+nr];

	if symmetric == False:
		
		f_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			j = N_fm-jj; # k_s wave-number, will be odd
			ind_j = (j-1)*nr; # Row ind

			f[ind_j:ind_j+nr] = D2.dot(g[ind_j:ind_j+nr]) -j*IR4.dot( (j+1)*g[ind_j:ind_j+nr] + 2.*f_e);
			f_e += g[ind_j:ind_j+nr];	

	return f;

#~~~~~~~ Validated up to here ~~~~~~~~~~

@njit(fastmath=True)
def Vecs_To_NX(PSI,T,C, aks,akc, N_fm,nr, symmetric = False):

	for jj in range(nr): # nr*O(N_fm*ln(N_fm)) Correct
		PSI[jj,:] = aks*PSI[jj,:]; 
		T[jj,:]   =	akc*T[jj,:];
		C[jj,:]   = akc*C[jj,:]; 	
		 
	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	N  = N_fm*nr;
	NX = np.zeros(3*N);
	if symmetric == True:
		
		# O(N_fm/2) Correct

		for ii in range(1,N_fm,2): 
			
			# a) psi parts
			ind_p = ii*nr;
			NX[ind_p:ind_p+nr] = PSI[:,ii];
		
		for ii in range(0,N_fm,2):
			
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

	sp = (nr, int( 3*(N_fm/2)) ); # ZERO-PADDED FOR DE-ALIASING !!!!!
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
			
			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ ???
			ind_p = ii*nr; 
			psi   = X_hat[ind_p:ind_p+nr];

			Dpsi_hat[:,ii]    = Dr.dot(psi);
			kDpsi_hat[:,ii]   = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine
					
			JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr]; # Sine -> Cosine

			omega_hat[:,ii]   = OMEGA[ind_p:ind_p+nr];
			komega_hat[:,ii]  = k_s*omega_hat[:,ii] # Sine -> Cosine 

		for ii in range(0,N_fm,2): # Cosine [0,N_fm-1]

			k_c = ii;     # [0,N_fm-1]

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

def NLIN_FX(X_hat,	inv_D,D,R,N_fm,nr,aks,akc, symmetric = False, kinetic = False):

	"""

	Compute the nonlinear terms by taking the: 

	∂_s X(r,s) -> -k_s*X or -k_c*X, polar derivatives

	∂_r X(r,s) -> D*X, radial derivatives 

	return F(X,X) a vetor same shape as X

	"""

	from scipy.fft import dct, idct, dst, idst
	
	N  = nr*N_fm; 
	if N_fm%2 != 0:
		print("\n\n Must choose an even number of Fourier modes N_fm = 2*N_fm !!! \n\n");
		sys.exit();

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	JPSI  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric)
	OMEGA = A2_SINE_R2(X_hat[0:N], N_fm,nr,D,R, symmetric);
	Dr    = D[1:-1,1:-1];
	JPSI  = np.ascontiguousarray(JPSI);
	OMEGA = np.ascontiguousarray(OMEGA);
	Dr    = np.ascontiguousarray(Dr);
	
	# 1) Compute derivatives & Transform to Nr x N_fm
	JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat = Derivatives(X_hat,JPSI,OMEGA, Dr,N_fm,nr, symmetric);
	
	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# psi, T,C 
	JT_psi = idct( JT_psi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	komega = idct(komega_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	kDpsi  = idct( kDpsi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	DT 	   = idct(DT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay 
	DC     = idct(DC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay

	# psi, T, C
	omega = idst( omega_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	Dpsi  = idst(  Dpsi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	kT 	  = idst(kT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift
	kC 	  = idst(kC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*omega) - (kDpsi*omega + Dpsi*komega);
	NJ_PSI_T = JT_psi*DT - Dpsi*kT;	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;

	# 4) Compute DCT and DST, un-pad, multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = dst(NJ_PSI__,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm];
	J_PSI_T_hat = dct(NJ_PSI_T,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm]; 
	J_PSI_C_hat = dct(NJ_PSI_C,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm];		

	if kinetic == True:

		KE = Kinetic_Energy(JT_psi,Dpsi, aks,akc, R,inv_D,N_fm,nr);
		return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	aks,akc,	N_fm,nr, symmetric), KE;
	else:
		return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	aks,akc,	N_fm,nr, symmetric);	



def Kinetic_Energy(Jψ,dr_ψ, aks,akc, R,inv_D,N_fm,nr):

	"""

	Compute the volume integrated kinetic energy


	"""
	from scipy.fft import dst

	IR2  = np.diag(1./(R[1:-1]**2));
	IR2  = np.ascontiguousarray(IR2);

	#u_r = IR2@Jψ;
	#u_θ = -IR@dr_ψ;
	#KE_rθ = abs(u_r)**2 + abs(u_θ)**2;

	KE_rθ = IR2@(Jψ**2) + abs(dr_ψ)**2;
	
	# Integrate in θ and take zero mode essentially the IP with 
	# KE = int_r1^r2 (1/2)int_0^π KE(r,θ) r^2 sin(θ) dr dθ

	KE_r       = np.zeros(len(R));
	KE_r[1:-1] = aks[0]*dst(KE_rθ,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0];

	# Multiply by r^2 and integrate w.r.t r
	#KE_r = R*R*KE_r;
	
	KE = inv_D[0,:].dot(KE_r[0:-1]);
	V  = 2.*abs(R[-1] - R[0]); # here we divide by 2 as thats what the volume integral gives

	return (1./V)*KE;

'''
Nr = 10; d=1.
D,R = cheb_radial(Nr,d)
nr = len(R[1:-1]); print("nr = ",nr,"\n")
N_fm = 2*20;

#print(A.toarray())
#import matplotlib.pyplot as plt 
#plt.spy(A);
#plt.show()

L = T0J_theta(R,N_fm,d);#.toarray();

g = np.random.randn(N_fm*nr);
#for k in range(0,N_fm,2):
#	g[k*nr:(k+1)*nr] = 0. 

f_anal = L.dot(g);
print(f_anal)

f_num = J_theta_RT(g,     nr,N_fm, symmetric = False)
print(f_num);

print(np.linalg.norm(f_anal - f_num,np.inf));
sys.exit()
'''

def NLIN_DFX(dv_hat,X_hat,	inv_D,D,R,N_fm,nr,aks,akc, symmetric = False):

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
	JT_psi = idct( JT_psi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	komega = idct(komega_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	kDpsi  = idct( kDpsi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	DT 	   = idct(DT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay 
	DC     = idct(DC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay

	# psi, T, C
	omega = idst( omega_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	Dpsi  = idst(  Dpsi_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	kT 	  = idst(kT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift
	kC 	  = idst(kC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift


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
	δJT_ψ = idct(δJT_ψ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	δkΩ   = idct(  δkΩ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	δkDψ  = idct( δkDψ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # needs shift
	δDT   = idct(  δDT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay 
	δDC   = idct(  δDC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay

	# psi, T, C
	δΩ  = idst(  δΩ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	δDψ = idst( δDψ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Projected okay
	δkT = idst( δkT_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift
	δkC = idst( δkC_hat,type=2,axis=-1,norm='ortho',overwrite_x=True) # Needs shift


	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*δΩ)   - (kDpsi*δΩ   + Dpsi*δkΩ);
	NJ_PSI__+= Dr@(δJT_ψ*omega) - (δkDψ*omega + δDψ*komega)
	NJ_PSI_T = (δJT_ψ*DT - δDψ*kT) + (JT_psi*δDT - Dpsi*δkT);	
	NJ_PSI_C = (δJT_ψ*DC - δDψ*kC) + (JT_psi*δDC - Dpsi*δkC);

	# 4) Compute DCT and DST, un-pad, multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = dst(NJ_PSI__,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm];
	J_PSI_T_hat = dct(NJ_PSI_T,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm]; 
	J_PSI_C_hat = dct(NJ_PSI_C,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0:N_fm];		

	
	return Vecs_To_NX(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	aks,akc,	N_fm,nr, symmetric);	




'''
#  ~~~~~~~~ Main Routines ~~~~~~~~~~~

def PRECOND(X_in,NX,		N_fm,nr,tau, N2_INV,A4_INV,D2,IR4, compiler='Fortran', symmetric = True):


	N = N_fm*nr; s = (3*N,1)
	if X_in.shape != s:
		#print "Cast X_in to shape ",X_in.shape
		X = np.zeros(s); X[:,0] = X_in;
		#print "Casted X to shape ",X.shape
	else:
		X = X_in
	

	if compiler == 'Fortran':

		import L_PARA

		# 1) Vorticity
		if symmetric == True:
		
			NX[0:N,0] = L_PARA.non_lin1.a4_bsub_sym_v2(N_fm,nr,A4_INV,D2,IR4,NX[0:N] )-X[0:N,0];  # F90 Version	
			#print "PSI Agree ",np.allclose(NX1[0:N],NX[0:N],rtol=1e-12);

			# 2) Temperature
			NX[N:2*N,0] = L_PARA.non_lin1.nab2_sym_v2(N_fm,nr,N2_INV,NX[N:2*N])-X[N:2*N,0]; # F90 Version
			#print "T Agree ",np.allclose(NX1[N:2*N,0],NX[N:2*N,0],rtol=1e-08);

			# 3) Concentration
			NX[2*N:3*N,0] = (1./tau)*L_PARA.non_lin1.nab2_sym_v2(N_fm,nr,N2_INV,NX[2*N:3*N])-X[2*N:3*N,0]; # F90 Version
			#print "C Agree ",np.allclose(NX1[2*N:3*N],NX[2*N:3*N],rtol=1e-12);
		
		else:
			
			NX[0:N,0] = L_PARA.non_lin1.a4_bsub_v2(N_fm,nr,A4_INV,D2,IR4,NX[0:N] )-X[0:N,0];  # F90 Version
			#print "PSI Agree ",np.allclose(NX1[0:N],NX[0:N],rtol=1e-12);

			# 2) Temperature
			NX[N:2*N,0] = L_PARA.non_lin1.nab2_v2(N_fm,nr,N2_INV,NX[N:2*N])-X[N:2*N,0]; # F90 Version
			#print "T Agree ",np.allclose(NX1[N:2*N,0],NX[N:2*N,0],rtol=1e-08);

			# 3) Concentration
			NX[2*N:3*N,0] = (1./tau)*L_PARA.non_lin1.nab2_v2(N_fm,nr,N2_INV,NX[2*N:3*N])-X[2*N:3*N,0]; # F90 Version
			#print "C Agree ",np.allclose(NX1[2*N:3*N],NX[2*N:3*N],rtol=1e-12);	

	else:	
		# 1) Vorticity
		NX1[0:N] = A4_BSub(NX[0:N],N_fm,nr,D,R);#-X[0:N] # <12%

		# 2) Temperature
		#NX1[N:2*N] = LINV_T_SP.dot(NX[N:2*N]) -X[N:2*N];
		NX1[N:2*N] = NAB2_BSub(NX[N:2*N],N_fm,nr,D,R);# -X[N:2*N];

		# 3) Concentration
		#NX1[2*N:3*N] = LINV_C_SP.dot(NX[2*N:3*N]) -X[2*N:3*N]; # 12%	
		NX1[2*N:3*N] = (1./tau)*NAB2_BSub(NX[2*N:3*N],N_fm,nr,D,R); #-X[2*N:3*N];

	return NX;

##
# Should be able to combine the following two functions
def NLIN_FX_V2(X_in,			D,R,N_fm,aks,akc,D2_SINE,IR4, compiler='Fortran', symmetric = True):



	import L_PARA

	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	Dr = D[1:-1,1:-1];

	# 1) Compute derivatives & Transform to Nr x N_fm
	nr = len(R[1:-1]); N = nr*N_fm; sp = (nr,3*(N_fm/2)); # ZERO-PADDED ALIASING !!!!!
	#~~~~~~~~~~~ Wave-number Space ~~~~~~~~~~~~~~


	if symmetric == True:

		# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
		JPSI = L_PARA.non_lin1.jt_sym(nr,N_fm,X[0:N]);
		OMEGA = L_PARA.non_lin1.a2_sine_sym(D2_SINE,IR4,nr,N_fm,X[0:N]);

		# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm  # O(nr^2*N_fm)
		Dpsi_hat,kDpsi_hat,JT_psi_hat,DJT_psi_hat,omega_hat,komega_hat,Domega_hat = L_PARA.non_lin1.deriv_psi_sym(X,JPSI,OMEGA,Dr,N_fm,nr);
		DT_hat,kT_hat,DC_hat,kC_hat = L_PARA.non_lin1.derivtc_sym(X,Dr,N_fm,nr);

	else:
		
		# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
		#JPSI = J_theta.dot(X[0:N]); 
		#OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
		JPSI = L_PARA.non_lin1.jt(nr,N_fm,X[0:N]);
		OMEGA = L_PARA.non_lin1.a2_sine(D2_SINE,IR4,nr,N_fm,X[0:N])
	    
	    # Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm  # O(nr^2*N_fm)
		Dpsi_hat,kDpsi_hat,JT_psi_hat,DJT_psi_hat,omega_hat,komega_hat,Domega_hat = L_PARA.non_lin1.deriv_psi(X,JPSI,OMEGA,Dr,N_fm,nr); #
		DT_hat,kT_hat,DC_hat,kC_hat = L_PARA.non_lin1.derivtc(X,Dr,N_fm,nr);
		
	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;

	# b) sine -> cosine
	kDpsi_hat[:,1:] = kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;
	
	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# a) ~~~ iDCT ~~~~
	# a) psi parts
	DJT_psi = idct(DJT_psi_hat,type=2,norm='ortho',axis=1) # Projected okay
	JT_psi = idct(JT_psi_hat,type=2,norm='ortho',axis=1) # Projected okay
	
	kDpsi = idct(kDpsi_hat,type=2,norm='ortho',axis=1) # needs shift
	komega = idct(komega_hat,type=2,norm='ortho',axis=1) # needs shift
	
	# b) T N C parts
	DT = idct(DT_hat,type=2,norm='ortho',axis=1) # Projected okay
	DC = idct(DC_hat,type=2,norm='ortho',axis=1) # Projected okay	

	# b) ~~~ iDST ~~~~	
	# a) psi parts
	omega = idst(omega_hat,type=2,norm='ortho',axis=1) # Projected okay
	Domega = idst(Domega_hat,type=2,norm='ortho',axis=1) # Projected okay
	Dpsi = idst(Dpsi_hat,type=2,norm='ortho',axis=1) # Projected okay

	# Cosine -> Sine, shift back with u[-1] = 0.0
	# b) T parts
	kT = idst(kT_hat,type=2,norm='ortho',axis=1) # Needs shift
	# c) C parts
	kC = idst(kC_hat,type=2,norm='ortho',axis=1) # Needs shift	
		

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NJ_PSI = DJT_psi*omega + JT_psi*Domega - (kDpsi*omega + Dpsi*komega);

	NJ_PSI_T = JT_psi*DT - Dpsi*kT;
	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;

	# 4) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI_hat = dst(NJ_PSI,type=2,norm='ortho',axis=1)[:,0:N_fm]; 
	J_PSI_T_hat = dct(NJ_PSI_T,type=2,norm='ortho',axis=1)[:,0:N_fm]; 
	J_PSI_C_hat = dct(NJ_PSI_C,type=2,norm='ortho',axis=1)[:,0:N_fm]; 	

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm , multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	#NX = np.zeros( (3*N,1));
	NX = L_PARA.non_lin1.reshape_nx(J_PSI_hat,J_PSI_T_hat,J_PSI_C_hat,aks,akc,N_fm,nr);

	# Pack computed terms DIM (nr,3/2*N_fm)
	X_sol = [JT_psi,DJT_psi,Dpsi,kDpsi,omega,Domega,komega,DT,DC,kT,kC];	
	
	return -NX, X_sol;

def LIN_DFX(X_in,X_sol, 	D,R,N_fm,aks,akc,D2_SINE,IR4, compiler='Fortran', symmetric = True):


	import L_PARA
	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	# 1) Compute derivatives & Transform to Nr x N_fm
	Dr = D[1:-1,1:-1]; nr = len(R[1:-1]); N = nr*N_fm; s = (3*N,1); # ZERO-PADDED ALIASING !!!!!
	
	#~~~~~~~~~~~ Wave-number Space ~~~~~~~~~~~~~~
	if X_in.shape != s:
		#print "Cast X_in to shape ",X_in.shape
		X = np.zeros(s); X[:,0] = X_in;
	else:
		X = X_in	
	
	if symmetric == True:
		# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
		#JPSI = J_theta.dot(X[0:N]); 
		#OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
		JPSI = L_PARA.non_lin1.jt_sym(nr,N_fm,X[0:N]);
		OMEGA = L_PARA.non_lin1.a2_sine_sym(D2_SINE,IR4,nr,N_fm,X[0:N])

		# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm  # O(nr^2*N_fm)
		Dpsi_hat,kDpsi_hat,JT_psi_hat,DJT_psi_hat,omega_hat,komega_hat,Domega_hat = L_PARA.non_lin1.deriv_psi_sym(X,JPSI,OMEGA,Dr,N_fm,nr);
		DT_hat,kT_hat,DC_hat,kC_hat = L_PARA.non_lin1.derivtc_sym(X,Dr,N_fm,nr);

	else:

		# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
		#JPSI = J_theta.dot(X[0:N]); 
		#OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
		JPSI = L_PARA.non_lin1.jt(nr,N_fm,X[0:N]);
		OMEGA = L_PARA.non_lin1.a2_sine(D2_SINE,IR4,nr,N_fm,X[0:N])

		# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm  # O(nr^2*N_fm)
		Dpsi_hat,kDpsi_hat,JT_psi_hat,DJT_psi_hat,omega_hat,komega_hat,Domega_hat = L_PARA.non_lin1.deriv_psi(X,JPSI,OMEGA,Dr,N_fm,nr);
		DT_hat,kT_hat,DC_hat,kC_hat = L_PARA.non_lin1.derivtc(X,Dr,N_fm,nr);


	#print "UN-SYM",komega_hat,"\n"; print "SYM",komega_hat_sym,"\n"
	#print "Check ",np.allclose(komega_hat_sym,komega_hat,rtol=1e-14)

	#print "UN-SYM",kT_hat,"\n"; print "SYM",kT_hat_sym,"\n"
	#print "Check ",np.allclose(kT_hat_sym,kT_hat,rtol=1e-14)

	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;
	# b) sine -> cosine
	kDpsi_hat[:,1:] = kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;
	
	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# a) ~~~ iDCT ~~~~
	# a) psi parts
	DJT_psi = idct(DJT_psi_hat,type=2,norm='ortho',axis=1) # Projected okay
	JT_psi = idct(JT_psi_hat,type=2,norm='ortho',axis=1) # Projected okay
	
	kDpsi = idct(kDpsi_hat,type=2,norm='ortho',axis=1) # needs shift
	komega = idct(komega_hat,type=2,norm='ortho',axis=1) # needs shift
	
	# b) T N C parts
	DT = idct(DT_hat,type=2,norm='ortho',axis=1) # Projected okay
	DC = idct(DC_hat,type=2,norm='ortho',axis=1) # Projected okay	

	# b) ~~~ iDST ~~~~	
	# a) psi parts
	omega = idst(omega_hat,type=2,norm='ortho',axis=1) # Projected okay
	Domega = idst(Domega_hat,type=2,norm='ortho',axis=1) # Projected okay
	Dpsi = idst(Dpsi_hat,type=2,norm='ortho',axis=1) # Projected okay

	# Cosine -> Sine, shift back with u[-1] = 0.0
	# b) T parts
	kT = idst(kT_hat,type=2,norm='ortho',axis=1) # Needs shift
	# c) C parts
	kC = idst(kC_hat,type=2,norm='ortho',axis=1) # Needs shift	

	# Un- Pack computed terms DIM (nr,3/2*N_fm)
	# X_sol = [JT_psi,DJT_psi,Dpsi,kDpsi,omega,Domega,komega,DT,DC,kT,kC];	
	JT_psi_b = X_sol[0]; DJT_psi_b = X_sol[1]; Dpsi_b = X_sol[2]; kDpsi_b = X_sol[3];
	omega_b = X_sol[4]; Domega_b = X_sol[5]; komega_b = X_sol[6];
	DT_b = X_sol[7]; DC_b = X_sol[8]; kT_b = X_sol[9]; kC_b = X_sol[10];

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	# Calc N(X,X') using _b for base state
	NJ_PSI = DJT_psi_b*omega + JT_psi_b*Domega - (kDpsi_b*omega + Dpsi_b*komega);
	NJ_PSI_T = JT_psi_b*DT - Dpsi_b*kT;
	NJ_PSI_C = JT_psi_b*DC - Dpsi_b*kC;

	# Calc N(X',X) & add it to N(X,X')
	NJ_PSI += DJT_psi*omega_b + JT_psi*Domega_b - (kDpsi*omega_b + Dpsi*komega_b);
	NJ_PSI_T += JT_psi*DT_b - Dpsi*kT_b;
	NJ_PSI_C += JT_psi*DC_b - Dpsi*kC_b;

	# 4) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI_hat = dst(NJ_PSI,type=2,norm='ortho',axis=1)[:,0:N_fm]; 
	J_PSI_T_hat = dct(NJ_PSI_T,type=2,norm='ortho',axis=1)[:,0:N_fm]; 
	J_PSI_C_hat = dct(NJ_PSI_C,type=2,norm='ortho',axis=1)[:,0:N_fm]; 	

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm , multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NX = np.zeros( (3*N,1));
	NX[:,0] = L_PARA.non_lin1.reshape_nx(J_PSI_hat,J_PSI_T_hat,J_PSI_C_hat,aks,akc,N_fm,nr)	

	return -NX; # DF_X(X')
##

def LIN_LX( X_in,Ra,		gr,DT0,N,Tau,Ra_s, compiler='Fortran', symmetric = True):
	
	import L_PARA

	s = (3*N,1);
	if X_in.shape != s:
		X = np.zeros(s); X[:,0] = X_in;
	else:
		X = X_in		
	
	nr   = DT0.shape[0]; 
	N_fm = N/nr; 
	LX   = np.zeros(s);
	#T0_PSI = np.zeros((N,1)); RAG = np.zeros((N,1)); 
	#RAG[:,0] = gr.dot(Ra*X[N:2*N,0] - Ra_s*X[2*N:3*N,0]); 
	#T0_PSI[:,0] = L_PARA.non_lin1.t0_jt(DT0,nr,N_fm,X[0:N,0]);
	
	# 1) Vorticity
	LX[0:N,0] = gr.dot(Ra*X[N:2*N,0] - Ra_s*X[2*N:3*N,0]); 

	
	# 2) Temperature
	if compiler == "Fortran":
		
		if symmetric == True:
			LX[N:2*N,0] = L_PARA.non_lin1.t0_jt_sym(DT0,nr,N_fm,X[0:N,0]);
		else:
			LX[N:2*N,0] = L_PARA.non_lin1.t0_jt(DT0,nr,N_fm,X[0:N,0]);
	else:
		
		LX[N:2*N,0] = T0J_theta_RT(DT0,nr,N_fm,X[0:N,0])		

	# 3) Concentration
	LX[2*N:3*N] = LX[N:2*N]

	return LX;


# ~~~~~~~ ****** # ~~~~~~~ ******  # ~~~~~~~ ******  # ~~~~~~~ ****** 

#Not sure if this is needed ??
def PRECOND_TSTEP_UNSURE(X_in, Nx, LINV_psi_SP, LINV_T_SP, LINV_C_SP):

	N = LINV_psi_SP.shape[0]
	s = (3*N,1)
	if X_in.shape != s:
		X = np.zeros(s); X[:,0] = X_in;
	else:
		X = X_in
	
	# 1) Vorticity
	NX[0:N] = LINV_psi_SP.dot(NX[0:N]) -X[0:N] # <12%

	# 2) Temperature
	#NX[N:2*N] += T0_PSI + NABLA2.dot(X[N:2*N]);
	NX[N:2*N] = LINV_T_SP.dot(NX[N:2*N]) -X[N:2*N]; # <12 ?%

	# 3) Concentration
	#NX[2*N:3*N] += T0_PSI +  Tau*NABLA2.dot(X[2*N:3*N]);
	NX[2*N:3*N] = LINV_C_SP.dot(NX[2*N:3*N]) -X[2*N:3*N]; # 12%	

	return NX;

# ~~~~~~~~~~~~ Interpolation functions ~~~~~~~~~~~~~~

def INTERP_RADIAL(R_n,R_o,X_0):

	nr_n = len(R_n)-2; #R_n = R_n[1:-1]; 
	
	nr = len(R_o)-2; #R_o = R_o[1:-1];
	X_o = np.zeros(nr);
	
	N_fm = len(X_0)//(3*nr);
	
	NN = 3*nr_n*N_fm; XX = np.zeros((NN,1));
	
	for l in range(N_fm):
		
		#print("Mode k",l)

		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		ind = nr*l;		
		for ii in range(nr):
			X_o[ii] =  X_0[ind+ii];
		# Pad adds zeros to the ends, evaluates in grid space
		PSI = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o));	
			
		ind_n = nr_n*l;
		# Polyvals to collocation space on new grid
		XX[ind_n:ind_n+nr_n,0] = np.polyval(PSI,R_n[1:-1])
		
		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		ind = N_fm*nr + nr*l;
		for ii in range(nr):
			X_o[ii] =  X_0[ind+ii];
		T = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o));

		ind_n = N_fm*nr_n + nr_n*l;
		XX[ind_n:ind_n + nr_n,0] = np.polyval(T,R_n[1:-1])
		
		# ~~~~ C ~~~~~~~~~~~~~~~~~~
		ind = 2*N_fm*nr + nr*l;
		for ii in range(nr):
			X_o[ii] =  X_0[ind+ii];
		OM = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o));
		ind_n = 2*N_fm*nr_n + nr_n*l;
		XX[ind_n:ind_n+nr_n,0] = np.polyval(OM,R_n[1:-1])
	
	return XX;

def INTERP_THETA_UP(N_fm,N_fm_o,X_0):

	nr = len(X_0)//(3*N_fm_o);
	print(nr)
	NN = 3*nr*N_fm; 
	print(NN)
	XX = np.zeros((NN,1));
	
	N = nr*N_fm; N_o = nr*N_fm_o
	#
	PSI_X = np.zeros((nr,N_fm)); T_X = np.zeros((nr,N_fm)); C_X = np.zeros((nr,N_fm));
	

	# 1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for k in range(N_fm_o):
		#print("k ",k)
		ind = k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		PSI_X[:,k] = X_0[ind:ind +nr,0] 
		
		ind = N_o + k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		T_X[:,k] = X_0[ind:ind +nr,0] 

		ind = 2*N_o + k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		C_X[:,k] = X_0[ind:ind +nr,0] 
		

	# 2) iDCT or iDST Interpolate onto a grid 
	PSI = idst(PSI_X,type=2,norm='ortho',axis=1) 
	T = idct(T_X,type=2,norm='ortho',axis=1) 
	C = idct(C_X,type=2,norm='ortho',axis=1) 


	#SI_X = np.zeros((nr,N_fm)); T_X = np.zeros((nr,N_fm)); C_X = np.zeros((nr,N_fm));

	# 3) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	PSI_hat = dst(PSI,type=2,norm='ortho',axis=1); #[:,0:N_fm]; #don't pad
	T_hat = dct(T,type=2,norm='ortho',axis=1); #[:,0:N_fm]; #don't pad
	C_hat = dct(C,type=2,norm='ortho',axis=1); #[:,0:N_fm]; #don't pad


	# 4) DCT or DST onto more polynomials
	for k in range(N_fm):
		
		ind = k*nr
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = PSI_hat[:,k];

		ind = N + k*nr
		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = T_hat[:,k];

		ind = 2*N + k*nr
		# ~~~~ C ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = C_hat[:,k];

	print("Interpolated")

	return XX;	

def INTERP_THETA_DOWN(N_fm,N_fm_o,X_0):

	nr = len(X_0)//(3*N_fm_o);
	
	NN = 3*nr*N_fm; XX = np.zeros((NN,1));
	
	N = nr*N_fm; N_o = nr*N_fm_o;
	
	PSI_X = np.zeros((nr,N_fm_o)); T_X = np.zeros((nr,N_fm_o)); C_X = np.zeros((nr,N_fm_o));
	

	# 1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for k in range(N_fm_o):
		#print("k ",k)
		ind = k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		PSI_X[:,k] = X_0[ind:ind +nr,0] 
		
		ind = N_o + k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		T_X[:,k] = X_0[ind:ind +nr,0] 

		ind = 2*N_o + k*nr;
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		C_X[:,k] = X_0[ind:ind +nr,0] 
		

	# 2) iDCT or iDST Interpolate onto a grid 
	PSI = idst(PSI_X,type=2,norm='ortho',axis=1) 
	T = idct(T_X,type=2,norm='ortho',axis=1) 
	C = idct(C_X,type=2,norm='ortho',axis=1) 


	#SI_X = np.zeros((nr,N_fm)); T_X = np.zeros((nr,N_fm)); C_X = np.zeros((nr,N_fm));

	# 3) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	PSI_hat = dst(PSI,type=2,norm='ortho',axis=1)[:,0:N_fm]; #don't pad
	T_hat = dct(T,type=2,norm='ortho',axis=1)[:,0:N_fm]; #don't pad
	C_hat = dct(C,type=2,norm='ortho',axis=1)[:,0:N_fm]; #don't pad


	# 4) DCT or DST onto more polynomials
	for k in range(N_fm):
		
		ind = k*nr
		# ~~~~ Psi ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = PSI_hat[:,k];

		ind = N + k*nr
		# ~~~~ T ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = T_hat[:,k];

		ind = 2*N + k*nr
		# ~~~~ C ~~~~~~~~~~~~~~~~~~
		XX[ind:ind +nr,0] = C_hat[:,k];

	print("Interpolated")

	return XX;		

#*~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~zw~~~~~~ *~~~~~~~~~~

# ~~~~~ Full Nr x N_fm blocks ~~~~

# ~~~~~ NABLA2 Cosine~~~~ Correct as of 16/09/22
def A2_theta_C(R,N_fm): # No 1/R^2
	
	# LAP2_theta Cosine-Basis  = r^2 \nabla^2 + A^2_\theta

	from scipy.sparse  import bmat
	from scipy.sparse  import diags
	
	nr = len(R[1:-1]); 
	IR = np.eye(nr); #diags( np.ones(nr),0,format="csr");  #

	AT = [];
	for j in range(N_fm): # [0,N_Fm -1]
		AT_j = [];
		for k in range(N_fm): # [0,N_Fm -1]
			
			if (k == j):
				AT_j.append(-k*(k + 1.0)*IR)
			elif (k > j) and ( (k+j)%2 == 0 ):
				AT_j.append(-k*2.0*IR)	
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
	s
	from scipy.sparse  import diags
	from scipy.sparse  import bmat

	R_1 = 1./d; 
	R_2 = (1. + d)/d;
	A_t = (R_1*R_2)/(R_1- R_2)

	IR = -A_t*diags( 1.0/(R[1:-1]**2),0,format="csr"); 
	
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

# ~~~~~ g(r)_d_theta ~~~~ not rechecked
def kGR(R,N_fm,d): # Correct
	
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
	
	IR = diags( 1.0/(R[1:-1]**2),0,format="csr");  
	
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

'''	