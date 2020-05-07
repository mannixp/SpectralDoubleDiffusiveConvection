#! /usr/bin/env python

'''#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------
'''
import numpy as np
from scipy.sparse import bmat, diags, block_diag, eye
import os, time

from numpy import asfortranarray
from scipy.io import FortranFile

import warnings
warnings.simplefilter('ignore', np.RankWarning)
print os.getcwd()

'''
#~~~~~~~~~~~~~~~~~~~~~~ Load all integral coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#Stringy = ['/home/mannixp/Dropbox/Leeds/Pseudo_DD/Integrated_Modes100/']
Stringy = ['/home/mannixp/Dropbox/Leeds/Integrated_Modes300/']
#Stringy = ['/home/ma/p/pm4615/Integrated_Modes/']
os.chdir("".join(Stringy))

NN = 30 + 1;
# PSI Coefficients
aPsi = np.load("aPsi.npy")[0:NN,0:NN,0:NN]; bPsi = np.load("bPsi.npy")[0:NN,0:NN,0:NN];
aPSI = np.asfortranarray(aPsi); bPSI = np.asfortranarray(bPsi);

# Temp/Solute Coefficients
aTt = np.load("aT.npy")[0:NN,0:NN,0:NN]; bTt = np.load("bT.npy")[0:NN,0:NN,0:NN];
aT = np.asfortranarray(aTt); bT = np.asfortranarray(bTt);


Stringy = ['/home/mannixp/Dropbox/Leeds/Pseudo_DD/']
#Stringy = ['/home/ma/p/pm4615/']
os.chdir("".join(Stringy))

aPSI.T.tofile("aPSI.bin");
bPSI.T.tofile("bPSI.bin");
aT.T.tofile("aT.bin");
bT.T.tofile("bT.bin");
sys.exit()
'''
'''
f = FortranFile('aPsi.unf', 'w');
f.write_record(aPSI);
f.close()

f = FortranFile('bPsi.unf', 'w');
f.write_record(bPSI);
f.close()


f = FortranFile('aT.unf', 'w');
f.write_record(aT);
f.close()

f = FortranFile('bT.unf', 'w');
f.write_record(bT);
f.close()
'''
'''
#print "Python aPSI[1,2,1] ",aPsi[:,:,1], "\n"
print "Ratio aPsi ",float(np.count_nonzero(aPsi[10,:,50])), "\n"
print "Ratio bPsi ",float(np.count_nonzero(bPsi[10,:,50])), "\n"
print "Ratio aT ",float(np.count_nonzero(aT[10,:,50])), "\n"
print "Ratio bT ",float(np.count_nonzero(bT[10,:,50])), "\n"
#print "Python aT[1,2,1] ",aTt[:,:,2], "\n"
#print "Python bT[1,2,1] ",bTt[3,4,1], "\n"
sys.exit()
'''

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
		print "Mode",l
		
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
	r_i = 1.0; r_o = 1.0+d;

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

# Build D^3
def D3_SP(D,r,l):
	
	r_i = r[-1]; r_o = r[0];
	b = -(r_i + r_o); c = r_i*r_o
	
	S = np.diag(1.0/((r[:]**2)+b*r[:]+c));
	S[0,0] = 0.0; S[-1,-1] = 0.0;

	# All matrices are for v!!
	D2 = np.matmul(D,D);
	D3 = np.matmul(D,D2);
	
	L = np.matmul( np.diag(r[:]**2 + b*r[:] + c), D3) + np.matmul( 3.0*np.diag(2.0*r[:] + b),D2) + 6.0*D;
	L1 = np.matmul(L,S); # (d/dr)^3
	
	L2 = -np.matmul(np.diag(2.0/r[:]),D2); 
	L3 = -l*(l+1.0)*np.matmul(np.diag(1.0/(r[:]**2)),D);
	L4 = 4.0*l*(l+1.0)*np.diag(1.0/(r[:]**3));

	A = L1 + L2 + L3 + L4;

	# Define the whole matrix as dim(R) then reduce by 2

	return A[1:-1,1:-1]

# Bouyancy term
def Bouyancy_SP(r,Ra):	
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
	return Ra*np.diag(1.0/(r**2))[1:-1,1:-1];

# Bounancy without divided by Ra
def Bouyancy_Ra_SP(r):
	
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

	return np.diag(1.0/(r**2))[1:-1,1:-1];

# Temperature gradient term if included
def temp_SP(r,l,d):
	
	# Parameters & Allocation
	A_T, B_T = -(1.0+d)/d, -1.0/d;
	
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
	MR2 = np.diag(r[1:-1]**2); Z0 = 0.0*MR2;
			
	ML_0 = np.bmat([[D2L,Z0,Z0],[Z0,MR2,Z0],[Z0,Z0,MR2]]);
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

'''# Build L1
def Ll_1_SP(r,l,Pr,d):
	
	s = len(r)-2;
	Z0 = np.zeros((s,s));

	# ~~~~~~~~~~~ Line 1 ~~~~~~~~~~~~~
	#D2D2l = Pr*Stokes_D2D2(R,l);
	RaG = Pr*Bouyancy_SP(r,1.0)
	# Zeros
		
	# ~~~~~~~~~~ Line 2 ~~~~~~~~~~~~~
	#TT = temp(R,l,d);
	#NABLA = laplacian(R,l);
	# Zeros

	# ~~~~~~~~~~~ Line 3 ~~~~~~~~~~~~~~~~
	# Zeros
	# Zeros
	#D2L = Pr*Stokes_D2(R,l);

	# Fit these blocks in
	if l == 0:
		Ll_0 = np.bmat([[Z0,Z0,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);	
	else:
		Ll_0 = np.bmat([[Z0,RaG,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);	
	return Ll_0
def L_1_SP(r,d,Pr,N_modes):
	
	s = 3*(len(r)-2);
	Z0 = np.zeros((s,s));

	# Define the full M_0 Matrix
	L_0 = [];
	for l in xrange(N_modes):
		L= [];
		for m in xrange(N_modes):
			if m == l: # Careful this is interpreted correctly
				L.append(Ll_1_SP(r,l,Pr,d));
			else:
				L.append(Z0);
		L_0.append(L)
	
	return np.bmat(L_0)
'''

# Built L0 with r^2
def Ll_0(D,r,d,l,Ra,Ra_s,Pr,Tau): # CORR
	
	s = len(r)-2; Z0 = np.zeros((s,s));
	MR2 = np.diag(r[1:-1]**2); A_T = -(1.0+d)/d;

	# ~~~~~~~~~~~ Line 1 ~~~~~~~~~~~~~
	D2D2l = Pr*np.matmul(MR2,Stokes_D2D2_SP(D,r,l));
	RaT = Pr*Ra*np.eye(s); #Bouyancy_SP(r,Ra) 
	RaC = -Pr*Ra_s*np.eye(s);
		
	# ~~~~~~~~~~ Line 2 ~~~~~~~~~~~~~
	x = A_T/(r[1:-1]**2)
	TT = (l*(l+1.0))*np.diag(x); #temp_SP(r,l,d);
	NABLA = np.matmul(MR2,laplacian_SP(D,r,l));
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

def L_0_SPAR(D,r,d,Tau,Pr,Ra,N_modes):
	
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

'''
def DF_Re1(Pr,R,N_modes,d,Re_2,Re_1): # Correct, CHANGED DIM

	# Taking Re_2 = 0

	nr = len(R)-2; s = (N_modes*3*nr,1)
	F = np.zeros(s); A = np.zeros(nr)
	
	alpha = 1.0+d; #eta = Re_2/Re_1;
	at = 1.0/( (alpha**3.0) - 1.0); bt = -(alpha**3.0)/( (alpha**3.0) - 1.0); # Correct constants

	for i in xrange(nr):
		A[i] = -2.0*( (at*bt)/(R[i+1]**2) + (bt*bt)/(R[i+1]**5) )*(2.0*Re_1);

	l = 2;	
	ind = l*3*nr;
	F[ind:ind+nr,0] = A[:]*(Pr**2);	
	return F

def DF_Re2(Pr,R,N_modes,d,Re_2,Re_1): # Correct, CHANGED DIM


	# Taking Re_1 = 0

	nr = len(R)-2; s = (N_modes*3*nr,1)
	F = np.zeros(s); A = np.zeros(nr)
	
	alpha = 1.0+d; #eta = Re_2/Re_1;
	at = -alpha/( (alpha**3.0) - 1.0); bt = alpha/( (alpha**3.0) - 1.0); # Correct constants

	for i in xrange(nr):
		A[i] = -2.0*( (at*bt)/(R[i+1]**2) + (bt*bt)/(R[i+1]**5) )*(2.0*Re_2);

	l = 2;	
	ind = l*3*nr;
	F[ind:ind+nr,0] = A[:]*(Pr**2);	
	return F
'''

#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     				3) Build Matrix Operator N_0			               #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 1) Execute kernprof -v -l 2Param_Cont.py
# 2) python -m line_profiler Time_Step.py.lprof

# Component Blocks, works for a Psi linearisation only
# l is the block
# m is the component it acts on
# n is the block row

# Corresponds to the block N1 
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#from scipy.sparse import diags

#import N1_F
#import N2_F
#print sys.exit()
def N_psi_T(Psi_0,l,m,n,D):

	D_B = D[1:-1,1:-1];

	PSI = np.diag(Psi_0); 
	PSI_D = np.matmul(PSI,D_B);
	D_PSI = np.diag(np.matmul(D_B,Psi_0));

	# create AT matrix
	#AT = aT[l,m,n]; #*MR2;
	# create BT matrix
	#BT = bT[l,m,n]; #*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return aT[l,m,n]*PSI_D + bT[l,m,n]*D_PSI;

#@profile
def N_psi_om(Omega_0,l,m,n,D,R):
	
	D_B = D[1:-1,1:-1];

	Omega = np.diag(Omega_0);
	Omega_D = np.matmul(Omega,D_B);

	# create A_OMEGA matrix
	#A_OMEGA = aPSIOM[l,m,n]*MR2;
	# create B_OMEGA matrix
	B_OMEGA = bPSIOM[l,m,n]*np.diag(1.0/R[1:-1]);

	return aPSIOM[l,m,n]*Omega_D + np.matmul(B_OMEGA,Omega);
	
#@profile
def N_psi_t_om(Psi_0,l,m,n,D,D2,D3):

	D_B = D[1:-1,1:-1];

	PSI = np.diag(Psi_0); 
	PSI_D = np.matmul(PSI,D_B);
	D_PSI = np.diag(np.matmul(D_B,Psi_0));

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# create A_PSI matrix
	#A_PSI = aPSI[l,m,n];#*MR2;
	PSI_D3M = np.matmul(PSI,D3[:,:,m]);
	DPSI_D2M = np.matmul(D_PSI,D2[:,:,m])
	# create B_PSI matrix
	#B_PSI = bPSI[l,m,n];#*MR2;

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	#AT = aT[l,m,n];#*MR2;
	# create BT matrix
	#BT = bT[l,m,n];#*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create A Omega matrix
	#A_OM = aOM[l,m,n];#*MR2;
	# create B Omega matrix
	#B_OM = bOM[l,m,n];#*MR2;

	return aPSI[l,m,n]*PSI_D3M + bPSI[l,m,n]*DPSI_D2M, aT[l,m,n]*PSI_D + bT[l,m,n]*D_PSI, aOM[l,m,n]*PSI_D + bOM[l,m,n]*D_PSI;
	
# Sub block
#@profile
def N_n(X,l,m,n,D,D2,D3,nr,R):

	NN = np.zeros((3*nr,3*nr)); #Z0 = np.zeros(N_PSI.shape);

	# Build and return the sub-block
	if l > 0:

		ind = l*3*nr;
		Psi_0 = np.zeros(nr); Psi_0[:] = X[ind:ind+nr,0].reshape(nr);
		# T_0 = np.zeros(nr);  # T_0[:] = X[ind+nr:ind+2*nr,0].reshape(nr); # Required ??
		Omega_0 = np.zeros(nr); Omega_0[:] = X[ind+2*nr:ind+3*nr,0].reshape(nr);

		# Obtain the matrix sub-blocks
		NN[0:nr,2*nr:3*nr] = N_psi_om(Omega_0,l,m,n,D,R);#,MR2,MR3); 
		NN[0:nr,0:nr],NN[nr:2*nr,nr:2*nr],NN[2*nr:3*nr,2*nr:3*nr] = N_psi_t_om(Psi_0,l,m,n,D,D2,D3)

		return NN;
	else:
		return NN;	

# Full block
#@profile

def N_Full(X,shape,N_modes,nr,D3,D2,D,R):

	N_F = np.zeros(shape)
	#N1_F = np.zeros(shape)
	ARGS_NF = [X,D3,D2,D[1:-1,1:-1],R[1:-1],aPSI,bPSI,aT,bT];
	N_F = N1_F.non_lin1.n1_full(*ARGS_NF)
	'''
	for l in xrange(N_modes):
		## Block mode l is fixed
		#N_F = []; # Define the full A_0 Matrix ????????
		#N_F = np.zeros(shape);
		for n in xrange(N_modes):
			# For each row n-changes as we become proportional to a different mode
			#A = []; 
			ind_n = n*3*nr
			for m in xrange(N_modes):
				
				ind_m = m*3*nr
				
				if (l+m+n)%2 == 0:
					if (abs(l-m) <= n) and (n <= abs(l+m)): 
						# Mode l, goes through the mode vector
						#A.append(N_n(X,l,m,n,D_B,dr,MR2,MR3,D2,D3,nr));  # 38%
						N_F[ind_n:ind_n+3*nr,ind_m:ind_m+3*nr] += N_n(X,l,m,n,D,D2,D3,nr,R); # 98%
						
						# Evaluate as a vector, everything in coloumn	
						#N_F[ind_n:ind_n+3*nr,0] += np.matmul(N_n(X,l,m,n,D,D2,D3,nr,R),X_m)

					#else:
					#	A.append(ZZ);	
				#else:
				#	A.append(ZZ);
			#N_F.append(A)
		
		#N = N + N_F; #np.bmat(N_F); 
		'''
	return 	N_F;

# Corresponds to the block N2 
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def N2_psi_T(T_0,l,m,n,D):

	D_B = D[1:-1,1:-1];

	D_T = np.diag(np.matmul(D_B,T_0));
	T_D = np.matmul(np.diag(T_0),D_B);
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	#AT = aT[l,m,n]; #*MR2;
	# create BT matrix
	#BT = bT[l,m,n]; #*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return aT[l,m,n]*D_T + bT[l,m,n]*T_D;

#@profile
def N2_psi_om(Omega_0,l,m,n,D,R):
	
	D_B = D[1:-1,1:-1];
	#Omega = np.diag(Omega_0); 
	D_Omega = np.diag(np.matmul(D_B,Omega_0)); # [D Omega_m]

	# create A_OMEGA matrix
	#A_OMEGA = aPSIOM[l,m,n]*MR2;
	# create B_OMEGA matrix
	B_OMEGA = bPSIOM[l,m,n]*np.diag(1.0/R[1:-1]);

	return aPSIOM[l,m,n]*D_Omega + np.matmul(B_OMEGA,np.diag(Omega_0));

#@profile
def N2_psi_t_om(Psi_0,T_0,Omega_0,l,m,n,D,D2,D3):

	#PSI = np.diag(Psi_0); 
	D_B = D[1:-1,1:-1];

	D_T = np.diag(np.matmul(D_B,T_0));
	T_D = np.matmul(np.diag(T_0),D_B);
	
	D_OM =  np.diag(np.matmul(D_B,Omega_0));
	OM_D = np.matmul(np.diag(Omega_0),D_B);
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# create A_PSI matrix
	A_PSI = aPSI[l,m,n];#*MR2;
	D3M_PSI = np.diag( np.matmul(D3[:,:,m],Psi_0) );
	# create B_PSI matrix
	B_PSI = bPSI[l,m,n];#*MR2;
	D2M_PSI = np.diag( np.matmul(D2[:,:,m],Psi_0) );

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	AT = aT[l,m,n];#*MR2;
	# create BT matrix
	BT = bT[l,m,n];#*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create A Omega matrix
	A_OM = aOM[l,m,n];#*MR2;
	# create B Omega matrix
	B_OM = bOM[l,m,n];#*MR2;

	return A_PSI*D3M_PSI + B_PSI*np.matmul(D2M_PSI,D_B), AT*D_T + BT*T_D, A_OM*D_OM + B_OM*OM_D;

# Sub block
#@profile
def N2_n(X,l,m,n,D,D2,D3,nr,R):

	NN = np.zeros((3*nr,3*nr)); #Z0 = np.zeros(N_PSI.shape);
	#D_B = D[1:-1,1:-1];
	# Build and return the sub-block
	if m==0:
		T_0 = np.zeros(nr); T_0[:] = X[nr:2*nr,0].reshape(nr);

		# OLD PYTHON  
		NN[nr:2*nr,0:nr] = N2_psi_T(T_0,l,m,n,D) 
		#NN[nr:2*nr,0:nr] = N_T.nonlin1.n2_psi_t(T_0,D_B,aT[l,m,n],bT[l,m,n]) # Current fortran
		return NN; #np.bmat([[Z0,Z0,Z0],[N_T,Z0,Z0],[Z0,Z0,Z0]]);
	else:
		ind = m*3*nr;

		Psi_0 = np.zeros(nr); Psi_0[:] = X[ind:ind+nr,0].reshape(nr); 
		T_0 = np.zeros(nr); T_0[:] = X[ind+nr:ind+2*nr,0].reshape(nr);
		Omega_0 = np.zeros(nr); Omega_0[:] = X[ind+2*nr:ind+3*nr,0].reshape(nr);


		# OLD PYTHON 
		NN[0:nr,2*nr:3*nr] = N2_psi_om(Omega_0,l,m,n,D,R);
		#NN[0:nr,2*nr:3*nr] = N_T.nonlin1.n2_psi_om(Omega_0,D_B,R[1:-1],aPSIOM[l,m,n],bPSIOM[l,m,n])

		'''
		if np.allclose(NN[0:nr,2*nr:3*nr],Test_Mat,rtol=1e-05) == False:
			
			print "OLD PYTHON",NN[0:nr,2*nr:3*nr],"\n"
			print "Current fortran", Test_Mat # Current fortran
			sys.exit()
		'''
		# OLD PYTHON  
		NN[0:nr,0:nr],NN[nr:2*nr,0:nr],NN[2*nr:3*nr,0:nr] = N2_psi_t_om(Psi_0,T_0,Omega_0,l,m,n,D,D2,D3)
		#args_OM = [Psi_0,T_0,Omega_0,D_B,D2[:,:,m],D3[:,:,m],aPSI[l,m,n],bPSI[l,m,n],aT[l,m,n],bT[l,m,n],aOM[l,m,n],bOM[l,m,n]];  
		#PP = N_T.nonlin1.n2_psi_t_om(*args_OM)
		#NN[0:nr,0:nr],NN[nr:2*nr,0:nr],NN[2*nr:3*nr,0:nr] = PP[0],PP[1],PP[2]
		return NN; 

# Full block
#@profile
def N2_Full(X,shape,N_modes,nr,D3,D2,D,R):
	
	N_F = np.zeros(shape)
	#N1_F = np.zeros(shape)
	ARGS_NF = [X,D3,D2,D[1:-1,1:-1],R[1:-1],aPSI,bPSI,aT,bT];
	N_F = N2_F.non_lin2.n2_full(*ARGS_NF)

	'''
	# Zero Block
	#ZZ = np.zeros((3*nr,3*nr));
	#ZZ = np.bmat([[Z0,Z0,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);
	for m in xrange(N_modes):
		## Block mode m is fixed
		#N_F = []; # Define the full A_0 Matrix ????????
		#N_F = np.zeros(shape);

		for n in xrange(N_modes):
			ind_n = n*3*nr
			# For each row n-changes as we become proportional to a different mode
			#A = [];
			for l in xrange(N_modes):
				ind_l = l*3*nr
				if (l+m+n)%2 == 0:
					if (abs(l-m) <= n) and (n <= abs(l+m)): 
						# Mode l, goes through the mode vector
						## OLD PYTHON VERSION
						#A.append(N2_n(X,l,m,n,D_B,dr,MR2,MR3,D2,D3,nr));  # 87%
						N_F[ind_n:ind_n+3*nr,ind_l:ind_l+3*nr] += N2_n(X,l,m,n,D,D2,D3,nr,R);

						## NEW FORTRAN VERSION
						#A_PSI, B_PSI = aPSI[l,m,n],bPSI[l,m,n]
						#A_T, B_T = aT[l,m,n],bT[l,m,n]
						#A_OM,B_OM = aOM[l,m,n],bOM[l,m,n]
						#A_PSIOM,B_PSIOM = aPSIOM[l,m,n],bPSIOM[l,m,n]


						#args_N2 = [X,D[1:-1,1:-1],D2[:,:,m],D3[:,:,m],R[1:-1],A_PSI,B_PSI,A_T,B_T,A_OM,B_OM,A_PSIOM,B_PSIOM,m];
						#print N_T.nonlin1.n2_n(*args_N2)
						#N_F[ind_n:ind_n+3*nr,ind_l:ind_l+3*nr] += N_T.nonlin1.n2_n(*args_N2)
						

					#else:
					#	A.append(ZZ);	
				#else:
				#	A.append(ZZ);		
			#N_F.append(A)
		
		#print "Finished Matrix N_F ",l,"\n"
		#N = N + N_F; # np.bmat(N_F); #8.5%
		'''
	return 	N_F;


'''
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     				3) Build Matrix Operator N_0 _____MR2			               #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 1) Execute kernprof -v -l 2Param_Cont.py
# 2) python -m line_profiler Time_Step.py.lprof

# Component Blocks, works for a Psi linearisation only
# l is the block
# m is the component it acts on
# n is the block row

# Corresponds to the block N1 
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#from scipy.sparse import diags

def N_psi_T(Psi_0,l,m,n,MR2,MR3,D2,D3):

	#D_B = D[1:-1,1:-1];

	PSI_D = np.matmul(np.diag(Psi_0),MR2);
	D_PSI = np.diag(np.matmul(MR2,Psi_0));

	# create AT matrix
	AT = aT[l,m,n]; #*MR2;
	# create BT matrix
	BT = bT[l,m,n]; #*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return AT*PSI_D + BT*D_PSI;

#@profile
def N_psi_om(Omega_0,l,m,n,MR2,MR3):
	
	#D_B = D[1:-1,1:-1];

	Omega = np.diag(Omega_0);
	Omega_D = np.matmul(Omega,MR2);

	# create A_OMEGA matrix
	A_OMEGA = aPSIOM[l,m,n]; #*MR2;
	# create B_OMEGA matrix
	B_OMEGA = bPSIOM[l,m,n]*MR3;

	return A_OMEGA*Omega_D + np.matmul(B_OMEGA,Omega);

#@profile
def N_psi_t_om(Psi_0,l,m,n,MR2,MR3,D2,D3):

	#D_B = D[1:-1,1:-1];

	PSI = np.diag(Psi_0); 
	PSI_D = np.matmul(PSI,MR2);
	D_PSI = np.diag(np.matmul(MR2,Psi_0));

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# create A_PSI matrix
	A_PSI = aPSI[l,m,n];#*MR2;
	PSI_D3M = np.matmul(PSI,D3[m]);
	# create B_PSI matrix
	B_PSI = bPSI[l,m,n];#*MR2;

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	AT = aT[l,m,n];#*MR2;
	# create BT matrix
	BT = bT[l,m,n];#*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create A Omega matrix
	A_OM = aOM[l,m,n];#*MR2;
	# create B Omega matrix
	B_OM = bOM[l,m,n];#*MR2;

	return A_PSI*PSI_D3M + B_PSI*np.matmul(D_PSI,D2[m]), AT*PSI_D + BT*D_PSI, A_OM*PSI_D + B_OM*D_PSI;

# Sub block
#@profile
def N_n(X,l,m,n,MR2,MR3,D2,D3,nr):

	NN = np.zeros((3*nr,3*nr)); #Z0 = np.zeros(N_PSI.shape);

	# Build and return the sub-block
	if l > 0:
		ind = l*3*nr;
		Psi_0 = np.zeros(nr); Psi_0[:] = X[ind:ind+nr,0].reshape(nr);
		# T_0 = np.zeros(nr);  # T_0[:] = X[ind+nr:ind+2*nr,0].reshape(nr); # Required ??
		Omega_0 = np.zeros(nr); Omega_0[:] = X[ind+2*nr:ind+3*nr,0].reshape(nr);


		# Obtain the matrix sub-blocks
		NN[0:nr,2*nr:3*nr] = N_psi_om(Omega_0,l,m,n,MR2,MR3); 
		NN[0:nr,0:nr],NN[nr:2*nr,nr:2*nr],NN[2*nr:3*nr,2*nr:3*nr] = N_psi_t_om(Psi_0,l,m,n,MR2,MR3,D2,D3)

		return NN;
	else:
		return NN;	

# Full block
#@profile
def N_Full(X,shape,N_modes,nr,MR2,MR3,D3,D2,D):

	N_F = np.zeros(shape)
	# Zero Block
	#ZZ = np.zeros((3*nr,3*nr));
	
	for l in xrange(N_modes):
		## Block mode l is fixed
		#N_F = []; # Define the full A_0 Matrix ????????
		#N_F = np.zeros(shape);
		for n in xrange(N_modes):
			# For each row n-changes as we become proportional to a different mode
			#A = []; 
			ind_n = n*3*nr
			for m in xrange(N_modes):
				
				ind_m = m*3*nr
				
				if (l+m+n)%2 == 0:
					if (abs(l-m) <= n) and (n <= abs(l+m)): 
						# Mode l, goes through the mode vector
						#A.append(N_n(X,l,m,n,D_B,dr,MR2,MR3,D2,D3,nr));  # 38%
						N_F[ind_n:ind_n+3*nr,ind_m:ind_m+3*nr] += N_n(X,l,m,n,MR2,MR3,D2,D3,nr); # 98%
					#else:
					#	A.append(ZZ);	
				#else:
				#	A.append(ZZ);
			#N_F.append(A)
		
		#N = N + N_F; #np.bmat(N_F); 

	return 	N_F;


# Corresponds to the block N2 
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def N2_psi_T(T_0,l,m,n,MR2,MR3,D2,D3):

	#D_B = D[1:-1,1:-1];

	D_T = np.diag(np.matmul(MR2,T_0));
	T_D = np.matmul(np.diag(T_0),MR2);
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	AT = aT[l,m,n]; #*MR2;
	# create BT matrix
	BT = bT[l,m,n]; #*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return AT*D_T + BT*T_D;

#@profile
def N2_psi_om(Omega_0,l,m,n,MR2,MR3):
	
	#D_B = D[1:-1,1:-1];
	#Omega = np.diag(Omega_0); 
	D_Omega = np.diag(np.matmul(MR2,Omega_0)); # [D Omega_m]

	# create A_OMEGA matrix
	A_OMEGA = aPSIOM[l,m,n]; #*MR2;
	# create B_OMEGA matrix
	B_OMEGA = bPSIOM[l,m,n]*MR3;

	return A_OMEGA*D_Omega + np.matmul(B_OMEGA,np.diag(Omega_0));

#@profile
def N2_psi_t_om(Psi_0,T_0,Omega_0,l,m,n,MR2,MR3,D2,D3):

	#PSI = np.diag(Psi_0); 
	#D_B = D[1:-1,1:-1];

	D_T = np.diag(np.matmul(MR2,T_0));
	T_D = np.matmul(np.diag(T_0),MR2);
	
	D_OM =  np.diag(np.matmul(MR2,Omega_0));
	OM_D = np.matmul(np.diag(Omega_0),MR2);
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# create A_PSI matrix
	A_PSI = aPSI[l,m,n];#*MR2;
	D3M_PSI = np.diag( np.matmul(D3[m],Psi_0) );
	# create B_PSI matrix
	B_PSI = bPSI[l,m,n];#*MR2;
	D2M_PSI = np.diag( np.matmul(D2[m],Psi_0) );

	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create AT matrix
	AT = aT[l,m,n];#*MR2;
	# create BT matrix
	BT = bT[l,m,n];#*MR2;
	
	# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create A Omega matrix
	A_OM = aOM[l,m,n];#*MR2;
	# create B Omega matrix
	B_OM = bOM[l,m,n];#*MR2;

	return A_PSI*D3M_PSI + B_PSI*np.matmul(D2M_PSI,MR2), AT*D_T + BT*T_D, A_OM*D_OM + B_OM*OM_D;

# Sub block
#@profile
def N2_n(X,l,m,n,MR2,MR3,D2,D3,nr):

	NN = np.zeros((3*nr,3*nr)); #Z0 = np.zeros(N_PSI.shape);

	# Build and return the sub-block
	if m==0:
		T_0 = np.zeros(nr); T_0[:] = X[nr:2*nr,0].reshape(nr);

		NN[nr:2*nr,0:nr] = N2_psi_T(T_0,l,m,n,MR2,MR3,D2,D3)
		return NN; #np.bmat([[Z0,Z0,Z0],[N_T,Z0,Z0],[Z0,Z0,Z0]]);
	else:
		ind = m*3*nr;

		Psi_0 = np.zeros(nr); Psi_0[:] = X[ind:ind+nr,0].reshape(nr); 
		T_0 = np.zeros(nr); T_0[:] = X[ind+nr:ind+2*nr,0].reshape(nr);
		Omega_0 = np.zeros(nr); Omega_0[:] = X[ind+2*nr:ind+3*nr,0].reshape(nr);

		NN[0:nr,2*nr:3*nr] = N2_psi_om(Omega_0,l,m,n,MR2,MR3); 
		NN[0:nr,0:nr],NN[nr:2*nr,0:nr],NN[2*nr:3*nr,0:nr] = N2_psi_t_om(Psi_0,T_0,Omega_0,l,m,n,MR2,MR3,D2,D3)

		return NN; #np.bmat([[N_PSI,Z0,N_PSI_OM],[N_T,Z0,Z0],[N_OM,Z0,Z0]]);

# Full block
#@profile
def N2_Full(X,shape,N_modes,nr,MR2,MR3,D3,D2,D):

	N_F = np.zeros(shape)
	# Zero Block
	#ZZ = np.zeros((3*nr,3*nr));
	#ZZ = np.bmat([[Z0,Z0,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);
	for m in xrange(N_modes):
		## Block mode m is fixed
		#N_F = []; # Define the full A_0 Matrix ????????
		#N_F = np.zeros(shape);

		for n in xrange(N_modes):
			ind_n = n*3*nr
			# For each row n-changes as we become proportional to a different mode
			#A = [];
			for l in xrange(N_modes):
				ind_l = l*3*nr
				if (l+m+n)%2 == 0:
					if (abs(l-m) <= n) and (n <= abs(l+m)): 
						# Mode l, goes through the mode vector
						#A.append(N2_n(X,l,m,n,D_B,dr,MR2,MR3,D2,D3,nr));  # 87%
						N_F[ind_n:ind_n+3*nr,ind_l:ind_l+3*nr] += N2_n(X,l,m,n,MR2,MR3,D2,D3,nr);
					#else:
					#	A.append(ZZ);	
				#else:
				#	A.append(ZZ);		
			#N_F.append(A)
		
		#print "Finished Matrix N_F ",l,"\n"
		#N = N + N_F; # np.bmat(N_F); #8.5%

	return 	N_F;

'''


#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

'''def Inner_Theta(N_Modes):
	beta1_KL = []; beta1_LK = [];
	beta2_KL = []; beta2_LK = [];
	beta3 = [];
	gamma1_LK = []; gamma1_KL = [];
	gamma2 = [];

	for l in xrange(N_Modes): # Block's row, corresponds to perturbation mode

		print "Mode l = ",l,"\n"
		print "\n"
		b1_KL = []; b1_LK = [];
		b2_KL = []; b2_LK = [];
		b3 = [];
		g1_LK = []; g1_KL = [];
		g2 = [];
		for k in xrange(N_Modes): # Coloumns in a row, Corresponds to base state modes

			print "l, Mode k = ",k
			# A) Compute all integrals, Required to be length N_theta but don't make all inner products, should save these speed
			b1_KL.append(BETA1_kl(k,l)); b1_LK.append(BETA1_kl(l,k));
			b2_KL.append(BETA2_kl(k,l)); b2_LK.append(BETA2_kl(l,k));
			b3.append(BETA3_kl(l,k));
			g1_LK.append(GAMMA1_kl(l,k)); g1_KL.append(GAMMA1_kl(k,l));
			g2.append(GAMMA2_kl(k,l));

		beta1_KL.append(b1_KL); beta1_LK.append(b1_LK);
		beta2_KL.append(b2_KL); beta2_LK.append(b2_LK);
		beta3.append(b3);
		gamma1_LK.append(g1_LK); gamma1_KL.append(g1_KL);
		gamma2.append(g2);			

	COEFFS = [beta1_KL,beta1_LK,beta2_KL,beta2_LK,beta3,gamma1_LK,gamma1_KL,gamma2];

	return COEFFS'''

#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     4) Some test plots     #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
'''
import math
def phi(Re_1,Re_2,Pr,d,r,Q): # Returns specific Angular Momentum

	alpha = 1.0+d; #eta = Re_2/Re_1;
	
	a = -(Re_2*alpha - Re_1)/( (alpha**3.0) - 1.0); b = -alpha*( Re_1*(alpha**2.0) - Re_2)/( (alpha**3.0) - 1.0);
	
	return (a*(r**2) + b/r)*(-np.sin(Q)**2)*Pr


def Angular(boo):
	OMEGA = np.zeros((Nr,Nth))
	for i in range(Nr):
		for j in range(Nth):    	
			OMEGA[i,j] = phi(Re_1,Re_2,Pr,d,R[i],theta[j])


	#-- Make Really Narrow Slice to form Axis---------------
	NN = 20
	# Using linspace so that the endpoint of 360 is included...
	azimuths = np.linspace(0,0.00001, NN)
	zeniths = np.linspace(0,4.0, NN )
	s = (NN,NN); ww = np.zeros(s); 


	if boo == True:
		fig1, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
		ax1.contourf(azimuths,zeniths,ww)

		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		try:
			p2 = ax1.contourf(theta, R, OMEGA,20) 
			ax1.contour(theta, R, OMEGA,20)		
		except ValueError:
			pass
		ax1.set_theta_zero_location("S")

		ax1.bar(math.pi, 0.0 )

		# Adjust the axis
		ax1.set_ylim(0.0,1.0+d)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		# ax.set_ylabel(r'\textbf{Radial position} (R)')
		ax1.set_xlabel(r'\textit{Polar Angle } ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')


		# Make space for title to clear
		plt.subplots_adjust(top=0.8)
		ax1.set_title(r'Specific Angular Momentum contours $\Omega$, $Re_2 = %s$, $Re_1 = %s$, $d= %s$'%(Re_2,Re_1,sigma), fontsize=16, va='bottom')

		cbar = plt.colorbar(p2)

		plt.savefig('AngularVelocityContours.eps', format='eps', dpi=1000)
		
		plt.show()

Pr = 1.0; sigma = 1.5; d = sigma;
Re_1 = 1.0; Re_2 = 0.0; # Re_1*((1.0+d)**2); 
Nr = 50; Nth = 100;
R = np.linspace(1.0,1.0+sigma,Nr); f = 1e-11
theta = np.linspace(f,np.pi-f,Nth)

Angular(True)
'''
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 					4) Single Mode Computation							   #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

'''mode_l = int(input("Single Mode Eig_Vals: Enter a mode number"));

if mode_l != 0:
	start_time = time.time()

	mode_l = 3;

	AA = Ll_0(R,mode_l,Pr,Ra,sigma);  # + NC_0(R,theta,N_Modes,OMEGA_NUM,PSI_NUM,T_NUM)
	BB = ML_0(R,mode_l);

	print "BB.shape ",BB.shape
	print "AA.shape ",AA.shape

	EIGS = scipy.linalg.eig(AA,b=BB)

	print "EigenVals ", EIGS[0][0:10], "\n"
	print "Eigenvectors ", EIGS[1].shape, "\n"

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))	

	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	n = 0;
	ind = n*3*(len(R)-2)
	EIG_PSI = np.zeros((len(R)-2,N_Modes)); #,N_Modes));
	EIG_T = np.zeros((len(R)-2,N_Modes)); #,N_Modes));
	EIG_OM = np.zeros((len(R)-2,N_Modes)); #,N_Modes));

	ii = 0; 
	print "Max Eigenval",np.max(EIGS[0].real)
	for jj in xrange(N_Modes):
		ind = (len(R)-2)
		nr = len(R)-2;
		EIG_PSI[:,jj] = EIGS[1][0:nr,jj].real;
		EIG_T[:,jj] = EIGS[1][nr:2*nr,jj].real;
		EIG_OM[:,jj] = EIGS[1][2*nr:3*nr,jj].real;

	plt.title(r'$\psi(r)$ Eigenvectors')
	plt.plot(R[1:len(R)-1],EIG_PSI[:,0],'r',label = r'$\lambda=0$')
	plt.plot(R[1:len(R)-1],EIG_PSI[:,1],'b',label = r'$\lambda =1$')
	plt.plot(R[1:len(R)-1],EIG_PSI[:,2],'y',label = r'$\lambda =2$')
	plt.plot(R[1:len(R)-1],EIG_PSI[:,3],'k',label = r'$\lambda =3$')
	#plt.plot(R[1:len(R)-1],EIG_PSI[:,4],'b',label = r'$\lambda =4$')
	plt.legend()
	plt.xlim([1.0,1.0+sigma])
	plt.show()


	plt.title(r'$T(r)$ Eigenvectors')
	plt.plot(R[1:len(R)-1],EIG_T[:,0],'r',label = r'$\lambda =0$')
	plt.plot(R[1:len(R)-1],EIG_T[:,1],'b',label = r'$\lambda =1$')
	plt.plot(R[1:len(R)-1],EIG_T[:,2],'y',label = r'$\lambda =2$')
	plt.plot(R[1:len(R)-1],EIG_T[:,3],'k',label = r'$\lambda =3$')
	#plt.plot(R[1:len(R)-1],EIG_T[:,4],'y',label = r'$\lambda =4$')
	plt.legend()
	plt.xlim([1.0,1.0+sigma])
	plt.show()

	plt.title(r'$\Omega(r)$ Eigenvectors')
	plt.plot(R[1:len(R)-1],EIG_OM[:,0],'r',label = r'$\lambda =0$')
	plt.plot(R[1:len(R)-1],EIG_OM[:,1],'b',label = r'$\lambda=1$')
	plt.plot(R[1:len(R)-1],EIG_OM[:,2],'y',label = r'$\lambda =2$')
	plt.plot(R[1:len(R)-1],EIG_OM[:,3],'k',label = r'$\lambda =3$')
	#plt.plot(R[1:len(R)-1],EIG_OM[:,4],'y',label = r'$\lambda =4$')
	plt.legend()
	plt.xlim([1.0,1.0+sigma])
	plt.show()'''

#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 					5) Multi-Mode Eigenvectors, for e					   #
#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
'''
def EIG_Main(Pr,Ra,sigma,R,N_modes,Re_2,Re_1,XX):

	start_time = time.time()

	L = L_0(Pr,Ra,sigma,R,N_modes); # + NC_0(R,theta,N_modes,OMEGA_NUM,PSI_NUM,T_NUM)
	#L_inv = np.linalg.inv(L)
	L_Omega  = L_OM(N_modes,R,sigma,Pr,Re_2,Re_1);
	L_Cor = L_COR(N_modes,R,sigma,Pr,Re_2,Re_1)

	M = M_0(R,N_modes);
	M_inv = np.linalg.inv(M)

	nr = len(R)-2; dr = R[1] - R[0];
	MR2 = np.diag(1.0/(R[1:-1]**2)); MR3 = np.diag(1.0/(R[1:-1]**3));
	D3 = []; D2 = []; D_B = D_b(R);
	for ii in xrange(N_modes):
		D3.append(D3_l(ii,R));
		D2.append(Stokes_D2(R,ii));

	# Recast X if required
	X = np.zeros((3*(len(R)-2)*N_modes,1))
	if len(X) > len(XX):
		X[0:len(XX),0] = XX[:,0].reshape(len(XX));
	elif len(X) < len(XX):
		X[:,0] = XX[0:len(X),0].reshape(len(X));	
	else:
		X = XX;	

	# Matrices of Nonlinear terms	
	N1_0 = N_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
	N2_0 = N2_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);

	A= L+ L_Omega + L_COR + N1_0 + N2_0

	A = np.matmul(M_inv,A);
	# Create symmetric hermitian ????
	#A = np.dot(A.T)
	end_time = time.time()
	print("Elapsed time Construction was %g seconds" % (end_time - start_time))

	start_time = time.time()
	
	
	eigenValues, eigenVectors = np.linalg.eig(A) # Don't Inlcude L_Omega !!
	#from scipy.sparse.linalg import eigsh
	#eigenValues, eigenVectors = eigsh(A,5, sigma=0, which='LM')

	# Re-arrange for the maximum eigenvalues & corresponding eigenvectors 
	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	print "EigenVals ",eigenValues[0:4]
	
	#EIGS = np.linalg.eigvals(A)
	#idx = EIGS.argsort()[::-1]   
	#eigenValues = EIGS[idx]
	#print "EigenVals ", eigenValues[0:5], "\n"
	
	end_time = time.time()
	print("Elapsed time Eval was %g seconds" % (end_time - start_time))	

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	plt.title(r'$\lambda$ Eigenvalues')
	plt.plot(eigenValues.imag,eigenValues.real,'bo',markerfacecolor='none') #marker='o',markerfacecolor='none', markeredgecolor='blue')
	plt.plot(np.linspace(0.5,-0.5,10),np.zeros(10),'k-',linewidth=1.0)
	plt.xlabel(r'$\Im \{ \lambda \}$',fontsize=20)
	plt.ylabel(r'$\Re \{ \lambda \}$',fontsize=20)
	plt.ylim([-100.0,20.0])
	#plt.xlim([20.0,-20.0])


	# ~~~~~~~~~~~~~~~~~ Create and Save files ~~~~~~~~~~~~~~~~####~~~~##
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	branch = 'SL'; # 'SL' 'RL'

	if Pr >= 1.0:
			STR = "".join([branch,'_SPECTRA_l2_Re',str(int(Re_1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_SPECTRA_l2_Re',str(int(Re_1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_SPECTRA_l2_Re',str(int(Re_1)),'_Pr001.eps']) 
	#plt.savefig(STR, format='eps', dpi=1200)
	plt.show()

	
	# Plot the least stable Eigenvector(s)	
	# ~~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ #~~~~~~~~~~
	X = np.zeros((3*(len(R)-2)*N_modes,1));
	
	#X[:,0] = X[:,0] + (eigenVectors[:,0].real)/np.linalg.norm(eigenVectors[:,0].real,2); # Used to initialize the time-stepping simulation
	X = (eigenVectors[:,0].real)/np.linalg.norm(eigenVectors[:,0].real,2);
	#for ii in xrange(3):
	#	#X[:,0] = X[:,0] + (eigenVectors[:,0].real)/np.linalg.norm(eigenVectors[:,0].real,2); # Used to initialize the time-stepping simulation
	#	X[:,0] = X[:,0] + (eigenVectors[:,ii].real)/np.linalg.norm(eigenVectors[:,ii].real,2);
	#print "Obtained Null Space X_1 \n"

	ii = 0;
	EIG_PSI = np.zeros((len(R)-2,N_modes)); 
	EIG_T = np.zeros((len(R)-2,N_modes)); 
	EIG_OM = np.zeros((len(R)-2,N_modes));
	for jj in xrange(N_modes):
		ind = jj*3*(len(R)-2);
		nr = len(R)-2;
		
		#EIG_PSI[:,jj] = eigenVectors[ind:ind+nr,ii].real 
		#EIG_T[:,jj] = eigenVectors[ind+nr:ind+2*nr,ii].real
		#EIG_OM[:,jj] = eigenVectors[ind+2*nr:ind+3*nr,ii].real 
		EIG_PSI[:,jj] = X[ind:ind+nr,0].reshape(nr);
		EIG_T[:,jj] = X[ind+nr:ind+2*nr,0].reshape(nr);
		EIG_OM[:,jj] = X[ind+2*nr:ind+3*nr,0].reshape(nr)

	#plt.title(r'$\psi(r)$ Eigenvectors')
	fig = plt.figure(figsize=(8,6))
	plt.title(r'Eigenvectors of A',fontsize=20)
	plt.ylabel(r'$\frac{X}{||X||}$',fontsize=20)
	plt.xlabel(r'$r$',fontsize=20)
	for l in xrange(20):#N_modes):
		#l = l+20;
		#if L2[0,l] >= 1e-04:
		if l == 0:
			
			plt.plot(R[1:len(R)-1],EIG_T[:,l],'-.',label = r'$T_{\ell =%s}$'%l)
		elif (l > 0): # or ( l == 4):
						
			plt.plot(R[1:len(R)-1],EIG_PSI[:,l],'*',label =r'$\psi_{\ell =%s}$'%l )
			plt.plot(R[1:len(R)-1],EIG_T[:,l],'^',label = r'$T_{\ell =%s}$'%l)
	
	# ~~~~~~~~~~~~~~~~~ Create and Save files ~~~~~~~~~~~~~~~~####~~~~##
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	plt.legend(loc=1)
	plt.xlim([1.0,1.0+sigma])
	if Pr >= 1.0:
		STR = "".join([branch,'_EIGFUN_l2_Re',str(int(Re_1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_EIGFUN_l2_Re',str(int(Re_1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_EIGFUN_l2_Re',str(int(Re_1)),'_Pr001.eps']) 	


	#plt.savefig(STR, format='eps', dpi=1200)
	#plt.savefig("".join(Some_String), format='eps', dpi=1200)
	plt.show()
	
	return X	


def Plot_Eigen(X,R):

	N_modes = len(X)/(3*(len(R)-2));
	ii = 0;
	EIG_PSI = np.zeros((len(R)-2,N_modes)); 
	EIG_T = np.zeros((len(R)-2,N_modes)); 
	EIG_OM = np.zeros((len(R)-2,N_modes));
	for jj in xrange(N_modes):
		ind = jj*3*(len(R)-2);
		nr = len(R)-2;
		 
		EIG_PSI[:,jj] = X[ind:ind+nr].reshape(nr);
		EIG_T[:,jj] = X[ind+nr:ind+2*nr].reshape(nr);
		EIG_OM[:,jj] = X[ind+2*nr:ind+3*nr].reshape(nr)

	#plt.title(r'$\psi(r)$ Eigenvectors')
	fig = plt.figure(figsize=(8,6))
	plt.title(r'Eigenvectors of A',fontsize=20)
	plt.ylabel(r'$\frac{X}{||X||}$',fontsize=20)
	plt.xlabel(r'$r$',fontsize=20)
	for l in xrange(4):
		#l = l+20;
		#if L2[0,l] >= 1e-04:
		if l == 0:
			plt.plot(R[1:len(R)-1],EIG_T[:,l],'-.',label = r'$T_{\ell =%s}$'%l)
		elif (l > 0): # or ( l == 4):
						
			plt.plot(R[1:len(R)-1],EIG_PSI[:,l],'*',label =r'$\psi_{\ell =%s}$'%l )
			plt.plot(R[1:len(R)-1],EIG_T[:,l],'^',label = r'$T_{\ell =%s}$'%l)
			plt.plot(R[1:len(R)-1],EIG_OM[:,l],'o',label = r'$\Omega_{\ell =%s}$'%l)
	
	# ~~~~~~~~~~~~~~~~~ Create and Save files ~~~~~~~~~~~~~~~~####~~~~##
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	plt.legend(loc=1)
	plt.xlim([1.0,R[-2]])
	#STR = "".join([branch,'_EIGFUN_l2_Re',str(int(Re_1)),'_Pr001.eps']) 	
	

	#plt.savefig(STR, format='eps', dpi=1200)
	#plt.savefig("".join(Some_String), format='eps', dpi=1200)
	plt.show()
'''