import numpy as np
from numba import njit
from numba.typed import List
from Transforms import IDCT, DCT, IDST, DST
import warnings
warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore')


def cheb_radial(N, d):
	
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


def Nabla2(D, r): 

	"""
	Build Operator - ∆ pre-mulitplied by r^2

	"""

	# r^2 T'' +2r T'
	D2 = np.diag(r[:]**2)@(D@D);
	RD = np.diag(2.0*r[:])@D;
	A  = D2 + RD

	# Leaving out the edges enforces the dircihlet b.c.s
	return A[1:-1,1:-1];


def Nabla4(D, r):
	
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


def R2(R, N_fm):
	
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


def kGR_RT(R, N_fm, d):
	
	"""
	Compute the operator g(r) ∂_s f(r,s,t) in spectral space 

	as g(r) k f_k(r,t) = R_1/r^2 k  k f_k(r,t)

	returns the sparse matrix which performs this operation
	"""

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


@njit(fastmath=True) 
def DT0_theta(g, dT0, N_fm, nr, symmetric=False): 
	
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
def A2_SINE(g,  D, R, N_fm, nr, symmetric=False): 

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


@njit(fastmath=True)
def J_theta_RT(g, nr, N_fm, symmetric=False):
	
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
def A2_SINE_R2(g, N_fm, nr, D, R, symmetric=False): 

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


@njit(fastmath=True)
def Vecs_to_X(PSI, T, C, N_fm, nr, symmetric=False):

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
def X_to_Vecs(X, N_fm, nr, symmetric=False):

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	N   = N_fm*nr;
	PSI = np.zeros((nr,N_fm));
	T   = np.zeros((nr,N_fm));
	C   = np.zeros((nr,N_fm));
	
	if symmetric == True:
		
		# O(N_fm/2) Correct

		for ii in range(1,N_fm,2): 
			
			#print("Row ii=%i, Sin(k_s*x) = %i"%(ii,ii+1))
			# a) psi parts
			ind_p = ii*nr;
			PSI[:,ii] = X[ind_p:ind_p+nr];
		
		for ii in range(0,N_fm,2): 	
			
			#print("Row ii=%i, Cos(k_c*x) = %i"%(ii,ii))
			# b) T parts
			ind_T = N + ii*nr; 
			T[:,ii] = X[ind_T:ind_T+nr];
			
			# c) C parts
			ind_C = 2*N + ii*nr;
			C[:,ii] = X[ind_C:ind_C+nr];
	
	elif symmetric == False:	
		
		# O(N_fm) Correct
		for ii in range(N_fm): 
			
			# a) psi parts
			ind_p = ii*nr;
			PSI[:,ii] = X[ind_p:ind_p+nr]
			
			# b) T parts
			ind_T = N + ii*nr; 
			T[:,ii] = X[ind_T:ind_T+nr]
			
			# c) C parts
			ind_C = 2*N + ii*nr;
			C[:,ii] = X[ind_C:ind_C+nr]
			
	return PSI,T,C;


@njit(fastmath=True)
def Derivatives(X_hat, JPSI, OMEGA, Dr, N_fm, nr, symmetric=False):

	sp = (nr, N_fm);
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

			Dpsi_hat[:,ii]    = Dr.dot(psi);		# Sine
			kDpsi_hat[:,ii]   = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine
					
		
			omega_hat[:,ii]   = OMEGA[ind_p:ind_p+nr];# Sine
			komega_hat[:,ii]  = k_s*omega_hat[:,ii]   # Sine -> Cosine 

		for ii in range(0,N_fm,2): # cosine [0,N_fm-1]
			
			k_c = ii;     # [0,N_fm-1]

			#print("Row ii=%i, Cosine(k_c*x) = %i"%(ii,k_c) )

			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~  # Correct
			ind_p = ii*nr; 
			JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr]; # Sine -> Cosine

			# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
			ind_T = N + ii*nr; 
			T 	  = X_hat[ind_T:ind_T+nr];

			DT_hat[:,ii] = Dr.dot(T);# Cosine
			kT_hat[:,ii] = -k_c*T;   # Cosine -> Sine

			# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
			ind_C = 2*N + ii*nr; 
			C 	  = X_hat[ind_C:ind_C+nr];

			DC_hat[:,ii] = Dr.dot(C);# Cosine
			kC_hat[:,ii] = -k_c*C;   # Cosine -> Sine

	elif symmetric == False:
		
		# O(nr^2*N_fm)
		for ii in range(N_fm):

			# Wavenumbers
			k_s = ii + 1; # [1,N_fm  ]
			k_c = ii;     # [0,N_fm-1]
			
			# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ # Correct
			ind_p = ii*nr; 
			psi   = X_hat[ind_p:ind_p+nr];

			Dpsi_hat[:,ii]    = Dr.dot(psi);		# Sine
			kDpsi_hat[:,ii]   = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine #
					
			JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr];  # Cosine

			omega_hat[:,ii]   = OMEGA[ind_p:ind_p+nr]; # Sine
			komega_hat[:,ii]  = k_s*omega_hat[:,ii];   # Sine -> Cosine 


			# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
			ind_T = N + ii*nr; 
			T 	  = X_hat[ind_T:ind_T+nr];

			DT_hat[:,ii] = Dr.dot(T);# Cosine
			kT_hat[:,ii] = -k_c*T;   # Cosine -> Sine

			# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
			ind_C = 2*N + ii*nr; 
			C 	  = X_hat[ind_C:ind_C+nr];

			DC_hat[:,ii] = Dr.dot(C);# Cosine
			kC_hat[:,ii] = -k_c*C;   # Cosine -> Sine

	# Convert Sine to sinusoids
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Dpsi_hat[:,1:]    =   Dpsi_hat[:,0:-1];  Dpsi_hat[:,0] = 0.0;
	kDpsi_hat[:,1:]   =  kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	omega_hat[: ,1:]  =  omega_hat[:,0:-1]; omega_hat[:,0] = 0.0;
	komega_hat[:,1:]  = komega_hat[:,0:-1];komega_hat[:,0] = 0.0;


	return JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat;


def NLIN_FX(X_hat, D, R, N_fm, nr, symmetric=False):

	"""

	Compute the nonlinear terms by taking the: 

	∂_s X(r,s) -> -k_s*X or -k_c*X, polar derivatives

	∂_r X(r,s) -> D*X, radial derivatives 

	return F(X,X) a vetor same shape as X

	"""

	N  = nr*N_fm; 
	if N_fm%2 != 0:
		raise ValueError('The number of Fourier modes is not even %d' %N_fm)

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	JPSI  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric)      # ~ cos(k_c*x)
	OMEGA = A2_SINE_R2(X_hat[0:N], N_fm,nr,D,R, symmetric); # ~ sin(k_s*x)
	Dr    = D[1:-1,1:-1];
	JPSI  = np.ascontiguousarray(JPSI);
	OMEGA = np.ascontiguousarray(OMEGA);
	Dr    = np.ascontiguousarray(Dr);
	
	# 1) Compute derivatives & Transform to Nr x N_fm
	JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat = Derivatives(X_hat,JPSI,OMEGA, Dr,N_fm,nr, symmetric);
	
	# # 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# # *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# psi, T,C 
	JT_psi = IDCT(JT_psi_hat,n = (3*N_fm)//2) 
	komega = IDCT(komega_hat,n = (3*N_fm)//2) 
	kDpsi  = IDCT( kDpsi_hat,n = (3*N_fm)//2) 
	DT 	   = IDCT(    DT_hat,n = (3*N_fm)//2)  
	DC     = IDCT(    DC_hat,n = (3*N_fm)//2) 

	# psi, T, C
	omega  = IDST( omega_hat,n = (3*N_fm)//2) 
	Dpsi   = IDST(  Dpsi_hat,n = (3*N_fm)//2) 
	kT 	   = IDST(    kT_hat,n = (3*N_fm)//2) 
	kC 	   = IDST(    kC_hat,n = (3*N_fm)//2)

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*omega) - (kDpsi*omega + Dpsi*komega);
	NJ_PSI_T = JT_psi*DT - Dpsi*kT;	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;

	# 4) Compute DCT and DST & un-pad
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = DST(NJ_PSI__,axis=-1)[:,0:N_fm];	
	J_PSI_T_hat = DCT(NJ_PSI_T,axis=-1)[:,0:N_fm];
	J_PSI_C_hat = DCT(NJ_PSI_C,axis=-1)[:,0:N_fm];

	# Convert from sinusoids back into my code's convention
	J_PSI___hat[:,0:-1] = J_PSI___hat[:,1:]; J_PSI___hat[:,-1] = 0.0;

	return Vecs_to_X(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	N_fm,nr, symmetric);


def NLIN_DFX(dv_hat ,X_hat, D, R, N_fm, nr, symmetric=False):

	"""

	Compute the Jacobian of the nonlinear terms F(X) by taking the: 

	∂_s X(r,s) -> -k_s*X or -k_c*X, polar derivatives

	∂_r X(r,s) -> D*X, radial derivatives 

	return DF(X)*dv = F(X,dv) + F(dv,X) a vetor same shape as X

	"""

	N  = nr*N_fm; 
	if N_fm%2 != 0:
		raise ValueError('The number of Fourier modes is not even %d' %N_fm)
	
	Dr    = D[1:-1,1:-1];
	Dr    = np.ascontiguousarray(Dr);

	# A)  Base state X terms

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	JPSI  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric)      # ~ cos(k_c*x)
	OMEGA = A2_SINE_R2(X_hat[0:N], N_fm,nr,D,R, symmetric); # ~ sin(k_s*x)	
	JPSI  = np.ascontiguousarray(JPSI);
	OMEGA = np.ascontiguousarray(OMEGA);

	# A.1) Compute derivatives & Transform to Nr x N_fm
	JT_psi_hat,kDpsi_hat,komega_hat,DT_hat,DC_hat,omega_hat,Dpsi_hat,kT_hat,kC_hat = Derivatives(X_hat,JPSI,OMEGA, Dr,N_fm,nr, symmetric);
	
	# A.2) ~~~~ Compute iDCT & iDST ~~~~~ 

	# psi, T,C 
	JT_psi = IDCT(JT_psi_hat,n = (3*N_fm)//2) 
	komega = IDCT(komega_hat,n = (3*N_fm)//2) 
	kDpsi  = IDCT( kDpsi_hat,n = (3*N_fm)//2) 
	DT 	   = IDCT(    DT_hat,n = (3*N_fm)//2)  
	DC     = IDCT(    DC_hat,n = (3*N_fm)//2) 

	# psi, T, C
	omega  = IDST( omega_hat,n = (3*N_fm)//2) 
	Dpsi   = IDST(  Dpsi_hat,n = (3*N_fm)//2) 
	kT 	   = IDST(    kT_hat,n = (3*N_fm)//2) 
	kC 	   = IDST(    kC_hat,n = (3*N_fm)//2)

	# B)  Perturbation ∆X terms

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	δJψ = J_theta_RT(dv_hat[0:N], nr,N_fm, symmetric)      # ~ cos(k_c*x)
	δΩ  = A2_SINE_R2(dv_hat[0:N], N_fm,nr,D,R, symmetric); # ~ sin(k_s*x)	
	δJψ = np.ascontiguousarray(δJψ);
	δΩ  = np.ascontiguousarray(δΩ);
	
	# B.1) Compute derivatives & Transform to Nr x N_fm
	δJT_ψ_hat,δkDψ_hat,δkΩ_hat,δDT_hat,δDC_hat,δΩ_hat,δDψ_hat,δkT_hat,δkC_hat = Derivatives(dv_hat,δJψ,δΩ, Dr,N_fm,nr, symmetric);

	# B.2) ~~~~ Compute iDCT & iDST ~~~~~ 

	# psi, T,C 
	δJT_ψ = IDCT(δJT_ψ_hat,n = (3*N_fm)//2) 
	δkΩ   = IDCT(  δkΩ_hat,n = (3*N_fm)//2) 
	δkDψ  = IDCT( δkDψ_hat,n = (3*N_fm)//2) 
	δDT   = IDCT(  δDT_hat,n = (3*N_fm)//2)  
	δDC   = IDCT(  δDC_hat,n = (3*N_fm)//2) 

	# psi, T, C
	δΩ    = IDST(   δΩ_hat,n = (3*N_fm)//2) 
	δDψ   = IDST(  δDψ_hat,n = (3*N_fm)//2) 
	δkT   = IDST(  δkT_hat,n = (3*N_fm)//2) 
	δkC   = IDST(  δkC_hat,n = (3*N_fm)//2) 


	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	NJ_PSI__ = Dr@(JT_psi*δΩ)   - (kDpsi*δΩ   + Dpsi*δkΩ);
	NJ_PSI__+= Dr@(δJT_ψ*omega) - (δkDψ*omega + δDψ*komega);
	NJ_PSI_T = (δJT_ψ*DT - δDψ*kT) + (JT_psi*δDT - Dpsi*δkT);	
	NJ_PSI_C = (δJT_ψ*DC - δDψ*kC) + (JT_psi*δDC - Dpsi*δkC);

	# 4) Compute DCT and DST & un-pad
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	J_PSI___hat = DST(NJ_PSI__,axis=-1)[:,0:N_fm];	
	J_PSI_T_hat = DCT(NJ_PSI_T,axis=-1)[:,0:N_fm];
	J_PSI_C_hat = DCT(NJ_PSI_C,axis=-1)[:,0:N_fm];

	# Convert from sinusoids back into my code's convention
	J_PSI___hat[:,0:-1] = J_PSI___hat[:,1:]; J_PSI___hat[:,-1] = 0.0;

	return Vecs_to_X(J_PSI___hat,J_PSI_T_hat,J_PSI_C_hat,	N_fm,nr, symmetric);


def INTERP_RADIAL(N_n, N_o, X_o, d):

	if N_n == N_o:
		return X_o;

	print('Interpolated in r from %d to %d'%(N_o,N_n),'\n')

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


def INTERP_THETAS(N_fm_n, N_fm_o, X_o):

	print('Interpolated in theta from %d to %d'%(N_fm_o,N_fm_n),'\n')
	if N_fm_n == N_fm_o:
		return X_o;

	from Transforms import DCT,DST,IDST,IDCT

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
	PSI_X = IDST(PSI_X) 
	T_X   = IDCT(T_X  ) 
	S_X   = IDCT(S_X  ) 


	# 3) Compute DCT and DST, un-pad De-ALIASING !!!!!
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	PSI_hat = DST(PSI_X,n=N_fm_n)
	T_hat   = DCT(T_X  ,n=N_fm_n)
	S_hat   = DCT(S_X  ,n=N_fm_n)

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


def NAB2_TSTEP_MATS(dt, N_fm, nr, D, R):

	N = nr*N_fm
	M0 = List()
	I = np.eye(nr)
	R2 = np.diag(R[1:-1]**2)
	R2_Nab2 = Nabla2(D,R)

	for jj in range(N_fm):		
		
		j = (N_fm - (jj + 1) )
		ind_j = j*nr
		
		bj = -j*(j + 1.0)
		A  = R2 - dt*(R2_Nab2 + bj*I)

		M0.append( np.ascontiguousarray(np.linalg.inv(A)) )
	
	return M0


@njit(fastmath=True)
def NAB2_BSub_TSTEP_V2(g, L_inv, N_fm, nr, dt, symmetric=False):
	
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

			if j < (N_fm - 2 ):
				ßk = -2.*(j + 2.);
				ßk = -dt*ßk;
				b += ßk*f[(j+2)*nr:(j+3)*nr];

			f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr] - b);

	# ~~~~~~~~~ Evens ~~~~~~~~~~~~			
	b = np.zeros(nr);
	for jj in range(1,N_fm,2):		
		
		j = (N_fm - (jj + 1) );
		ind_j = j*nr;
		
		if j < (N_fm - 2 ):
			ßk = -2.*(j + 2.);
			ßk = -dt*ßk;
			b += ßk*f[(j+2)*nr:(j+3)*nr];
		
		if j == 0:	
			f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr] - 0.5*b);	
		else:
			f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr] - b);
	

	return f;


def A4_TSTEP_MATS(dt, N_fm, nr, D, R):
	
	M0 = List()

	D4  = Nabla4(D, R)
	IR4 = np.diag( 1.0/(R[1:-1]**4) )

	IR2 = np.diag( 1.0/(R**2) ) 
	IR  = np.diag(1.0/R)
	D_sq= D@D 
	D2  = np.matmul(IR2, 2*D_sq - 4*IR@D + 6*IR2 )[1:-1,1:-1] 

	A2  = D_sq[1:-1,1:-1] 
	IR2 = IR2[1:-1,1:-1]

	for jj in range(N_fm):

		row = N_fm - (jj + 1)
		ind_j = row * nr
		j = N_fm - jj
		bj = -j*(j + 1)
		L1 = D2 + bj*IR4 
		L = (A2 + bj*IR2) - dt*(D4 + bj*L1)

		M0.append( np.ascontiguousarray(np.linalg.inv(L)) )

	return M0


@njit(fastmath=True)
def A4_BSub_TSTEP_V2(g, L_inv, D, R, N_fm, nr, dt, symmetric=False):

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

	IR2 = np.diag( 1.0/(R**2) ) 
	IR  = np.diag(1.0/R)
	D_sq= D@D 
	D2  = np.ascontiguousarray( (IR2@(2*D_sq - 4*IR@D + 6*IR2 ))[1:-1,1:-1] )

	IR2 = np.ascontiguousarray( IR2[1:-1,1:-1] )
	IR4 = np.ascontiguousarray( np.diag( 1.0/(R[1:-1]**4) ) )

	N = nr*N_fm; f = np.zeros(N) 
	
	
	# ~~~~~~~~~~~~~~~ EVENS ~~~~~~~~~~~~~~~~~~~~
	f_e = np.zeros(nr); bf_e = np.zeros(nr); 
	for jj in range(0,N_fm,2):

		row = N_fm - (jj + 1) 
		ind_j = row*nr
		j = N_fm-jj
		bj = -j*(j + 1)
		bjt = -2*j
		
		L1 = D2 + bj*IR4 

		if row < (N_fm - 2 ):
		
			f_e += f[(row+2)*nr:(row+3)*nr];

			# Add time component
			b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
			
			f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr]+b_test);   #O(Nr^3 N_theta)


			# Add sums after to get +2 lag
			bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

		else:
			f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr]);
			bf_e += bj*f[ind_j:ind_j+nr];

	if symmetric == False:
		# ~~~~~~~~~~~~~~~ ODDS ~~~~~~~~~~~~~~~~~~~~		
		f_e = np.zeros(nr); bf_e = np.zeros(nr); 
		for jj in range(1,N_fm,2):

			row = N_fm - (jj + 1)
			ind_j = row*nr
			j = N_fm-jj
			bj = -j*(j + 1)
			bjt = -2*j
			L1 = D2 + bj*IR4
			
			if row < (N_fm - 2 ):
			
				f_e += f[(row+2)*nr:(row+3)*nr];

				# Add time component
				b_test = dt*bjt*( L1.dot( f_e ) + IR4.dot( bf_e) ) - bjt*IR2.dot(f_e);
				
				f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr]+b_test);   #O(Nr^3 N_theta)

				# Add sums after to get +2 lag
				bf_e += bj*f[ind_j:ind_j+nr] + bjt*f_e;

			else:

				f[ind_j:ind_j+nr] = L_inv[jj]@(g[ind_j:ind_j+nr]);
				bf_e += bj*f[ind_j:ind_j+nr];

	return f;
