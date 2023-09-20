import numpy as np
from scipy.sparse import bmat, diags

import warnings
warnings.simplefilter('ignore', np.RankWarning)


def INTERP_SPEC(R_n,R_o,X_0):

	nr_n = len(R_n)-2; #R_n = R_n[1:-1]; 
	nr = len(R_o)-2; #R_o = R_o[1:-1];
	
	X_o = np.zeros(nr);
	
	N_modes = len(X_0)/(3*nr);
	NN = 3*nr_n*N_modes;
	XX = np.zeros((NN,1));
	for l in xrange(N_modes):
		ind_n = 3*nr_n*l; 
		ind = 3*nr*l;
		print("Mode",l)
		
		for ii in xrange(nr):
			X_o[ii] =  X_0[ind+ii];
		
		PSI = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o))
		XX[ind_n:ind_n+nr_n,0] = np.polyval(PSI,R_n[1:-1])
		
		for ii in xrange(nr):
			X_o[ii] =  X_0[ind+nr+ii];
		T = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o))
		XX[ind_n+nr_n:ind_n+2*nr_n,0] = np.polyval(T,R_n[1:-1])
		
		for ii in xrange(nr):
			X_o[ii] =  X_0[ind+2*nr+ii];
		OM = np.polyfit(R_o,np.pad(X_o,(1,1),'constant'),len(R_o))
		XX[ind_n+2*nr_n:ind_n+3*nr_n,0] = np.polyval(OM,R_n[1:-1])
	
	return XX;

#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     1) Build Spectral Operator Matrices                                       #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def cheb_radial(N,d):
	r_i = 1.0/d; r_o = (1.0+d)/d;

	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		x = 0.5*(r_o + r_i) + 0.5*(r_o-r_i)*x; # Transform to radial

		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	
	return D, x.reshape(N+1);

# Build Laplacian \nabla^2 operator
def laplacian_SP(D,r,l): 
	
	# D,r = cheb_radial(N,d); # Returns len N+1 matrix
	D2 = np.matmul(D,D);
	RD = np.matmul( np.diag(2.0/r[:]), D);
	S = l*(l+1.0)*np.diag(1.0/(r[:]**2));
	A = D2 + RD - S;

	return A[1:-1,1:-1]

# Build Stokes D^2 operator
def Stokes_D2_SP(D,r,l): 
	
	D2 = np.matmul(D,D);
	S = l*(l+1.0)*np.diag(1.0/(r[:]**2));
	A = D2 - S;
	
	# Define the whole matrix as dim(R) then reduce by 2

	return A[1:-1,1:-1]

# Build Stokes D^2D^2 operator
def Stokes_D2D2_SP(D,r,l): 
	
	# D,r = cheb_radial(N,d); # Returns len N+1 matrix
	I = np.ones(len(r));
	r_i = r[-1]; r_o = r[0];
	b = -(r_i + r_o); c = r_i*r_o
	
	S = np.diag(1.0/((r[:]**2)+b*r[:]+c));
	S[0,0] = 0.0; S[-1,-1] = 0.0;

	# All matrices are for v!!
	D2 = np.matmul(D,D);
	D3 = np.matmul(D,D2);
	D4 = np.matmul(D2,D2);
	
	L = np.matmul( np.diag(r[:]**2 + b*r[:] + c), D4) + np.matmul( 4.0*np.diag(2.0*r[:] + b),D3) + 12.0*D2;
	L1 = np.matmul(L,S); # (d/dr)^4
	
	L2 = -2.0*l*(l+1.0)*np.matmul(np.diag(1.0/(r[:]**2)),D2); 
	L3 = 4.0*l*(l+1.0)*np.matmul(np.diag(1.0/(r[:]**3)),D);
	L4 = ( (l*(l+1.0))**2 - 6.0*l*(l+1.0) )*np.diag(1.0/(r[:]**4));

	A = L1 + L2 + L3 + L4;

	return A[1:-1,1:-1]

# Bouyancy term
def Bouyancy_SP(r,Ra,d):	
	'''
	# Parameters
	nr= len(R); #-2
	dr= R[1] - R[0] 
	B = np.zeros((nr-2,nr-2))

	for jj in xrange(nr-2):
		B[jj,jj] = Ra*(1.0/(R[jj+1]**2));

	Nr = nr-2; D = np.zeros(Nr);         
	for ii in xrange(Nr):
		D[ii] = B[ii,ii];

	# Pentadiagonal Matrix		
	data = [D]; diags = [0] 
	AA = sparse.diags(data, diags)
	A = csc_matrix(AA);
	'''
	R_1 = 1./d
	return Ra*np.diag((R_1**2)/(r**2))[1:-1,1:-1];

# Bounancy without divided by Ra
def Bouyancy_Ra_SP(r,d):
	
	'''
	# Parameters
	nr= len(R)-2
	dr= R[1] - R[0] 
	B = np.zeros((nr,nr))

	for jj in xrange(nr):
		B[jj,jj] = (1.0/(R[jj+1]**2));


	Nr = nr-2; D = np.zeros(Nr);         
	for ii in xrange(Nr):
		D[ii] = B[ii,ii];

	# Pentadiagonal Matrix		
	data = [D]; diags = [0] 
	AA = sparse.diags(data, diags)
	A = csc_matrix(AA);	
	'''

	R_1 = 1./d
	return np.diag((R_1**2)/(r**2))[1:-1,1:-1];

# Temperature gradient term if included
def temp_SP(r,l,d):
	
	# Parameters & Allocation
	R_1 = 1./d; R_2 = (1. + d)/d;
	A_T = (R_1*R_2)/(R_1- R_2)
	
	'''
	B = np.zeros((len(R)-2,len(R)-2));

	for i in xrange(len(R)-2):
		B[i,i] = ( (l*(l+1.0))/(R[i+1]**2) )*( A_T/(R[i+1]**2) ) 

	Nr = nr-2; D = np.zeros(Nr);         
	for ii in xrange(Nr):
		D[ii] = B[ii,ii];

	# Pentadiagonal Matrix		
	data = [D]; diags = [0] 
	AA = sparse.diags(data, diags)
	A = csc_matrix(AA);	
	'''

	B1 = (l*(l+1.0))*np.diag(1.0/(r[:]**2));
	B2 = np.diag(A_T/(r[:]**2));

	return np.matmul(B1,B2)[1:-1,1:-1];

#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     2) Build Linear Blocks for Matrices, & Blocks  M_0,L_0               #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Built M with r^2
def ML_0(D,r,l): # CORR
	
	# All Nr-1 * Nr-1
	D2L = Stokes_D2_SP(D,r,l);
	#MR2 = np.diag(r[1:-1]**2); Z0 = 0.0*MR2;
	I = np.eye(len(r[1:-1])); Z0 = 0.0*I;	
	ML_0 = np.bmat([[D2L,Z0,Z0],[Z0,I,Z0],[Z0,Z0,I]]);
	#ML_0 = bmat([[np.matmul(MR2,D2L),None,None],[None,MR2,None],[None,None,MR2]]);
	return ML_0

# Built M with r^2
def ML_0_SPAR(D,r,l): # CORR
	
	# All Nr-1 * Nr-1
	D2L = Stokes_D2_SP(D,r,l);
	MR2 = np.diag(r[1:-1]**2); #Z0 = 0.0*MR2;
	#I = np.eye(len(r)-2); Z0 = 0.0*I;
	'''if l == 0:
		ML_0 = np.bmat([[D2L,Z0,Z0],[Z0,I,Z0],[Z0,Z0,I]]);
	else:	
		ML_0 = np.bmat([[D2L,Z0,Z0],[Z0,I,Z0],[Z0,Z0,I]]); '''
		
	return bmat([[np.matmul(MR2,D2L),None,None],[None,MR2,None],[None,None,MR2]]);

def M_0_SPAR(D,r,N_modes):
	
	#s = 3*(len(r)-2); Z0 = np.zeros((s,s));
	# Define the full M_0 Matrix
	M_0 = [];
	for ii in xrange(N_modes):
		M = []; # Declare an Empty row
		for jj in xrange(N_modes):
			if jj == ii:
				M.append(ML_0_SPAR(D,r,ii));
			else:
				M.append(None);
		M_0.append(M) # Fills M_0 with a new row
	
	return bmat(M_0)			

# Built L0 with r^2
def Ll_0(D,r,d,l,Ra,Ra_s,Pr,Tau): # CORR
	
	s = len(r)-2; Z0 = np.zeros((s,s));

	# ~~~~~~~~~~~ Line 1 ~~~~~~~~~~~~~
	D2D2l = Pr*Stokes_D2D2_SP(D,r,l);
	RaT = Pr*Ra*Bouyancy_Ra_SP(r,d) 
	RaC = -Pr*Ra_s*Bouyancy_Ra_SP(r,d) 
		
	# ~~~~~~~~~~ Line 2 ~~~~~~~~~~~~~
	#x = A_T/(r[1:-1]**2)
	TT = temp_SP(r,l,d);
	NABLA = laplacian_SP(D,r,l);
	# Zeros

	# ~~~~~~~~~~~ Line 3 ~~~~~~~~~~~~~~~~
	# TT_C
	# Zeros
	#NablaC = Tau*Nabla

	# Fit these blocks in
	if l == 0:
		Ll_0 = np.bmat([[D2D2l,Z0,Z0],[Z0,NABLA,Z0],[Z0,Z0,Tau*NABLA]]);
	else:
		Ll_0 = np.bmat([[D2D2l,RaT,RaC],[TT,NABLA,Z0],[TT,Z0,Tau*NABLA]]);	

	return Ll_0

def L_0(D,r,d,Ra,Ra_s,Pr,Tau,N_modes):
	
	s = 3*(len(r)-2);
	Z0 = np.zeros((s,s));

	# Define the full M_0 Matrix
	L_0 = [];
	for l in xrange(N_modes):
		L= [];
		for m in xrange(N_modes):
			if m == l: # Careful this is interpreted correctly
				L.append(Ll_0(D,r,d,l,Ra,Ra_s,Pr,Tau));
			else:
				L.append(Z0);

		L_0.append(L)
	
	return np.bmat(L_0)				

# ~~~~~~ SPARSE VERSIONS ~~~~~~~~~~~~~
def Ll_0_SPAR(D,r,d,l,Ra,Ra_s,Pr,Tau): # CORR
	
	s = len(r)-2; #Z0 = np.zeros((s,s));
	MR2 = np.diag(r[1:-1]**2); 
	A_T = -(1.0+d)/d; x = A_T/(r[1:-1]**2)

	# ~~~~~~~~~~~ Line 1 ~~~~~~~~~~~~~
	D2D2l = Pr*np.matmul(MR2,Stokes_D2D2_SP(D,r,l));
	RaG_T = Pr*Ra*diags(np.ones(s),offsets=0)
	RaG_C = -Pr*Ra_s*diags(np.ones(s),offsets=0)
		
	# ~~~~~~~~~~ Line 2 ~~~~~~~~~~~~~
	TT_T = (l*(l+1.0))*diags(x,offsets=0)
	NABLA = np.matmul(MR2,laplacian_SP(D,r,l));
	# Zeros

	# ~~~~~~~~~~~ Line 3 ~~~~~~~~~~~~~~~~
	#TT_C = (l*(l+1.0))*diags(x,offsets=0)
	# Zeros
	# NABLA = Tau*Nabla

	# Fit these blocks in
	if l == 0:
		Ll_s = bmat([[D2D2l,None,None],[None,NABLA,None],[None,None,Tau*NABLA]])
	else:
		Ll_s = bmat([[D2D2l,RaG_T,RaG_C],[TT_T,NABLA,None],[TT_T,None,Tau*NABLA]])
		
	return Ll_s

def L_0_SPAR(D,r,d,Tau,Pr,Ra,Ra_s,N_modes):
	
	#s = 3*(len(r)-2);
	#Z0 = np.zeros((s,s));

	# Define the full M_0 Matrix
	L_0 = [];
	for l in xrange(N_modes):
		L= [];
		for m in xrange(N_modes):
			if m == l: # Careful this is interpreted correctly
				L.append(Ll_0_SPAR(D,r,d,l,Ra,Ra_s,Pr,Tau));
			else:
				L.append(None);
				
		L_0.append(L)
	
	return bmat(L_0)	






