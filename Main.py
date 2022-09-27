#!/usr/bin/env python

from numba import njit
import numpy as np
import matplotlib.pyplot as plt

import sys, os, time, warnings, h5py
warnings.simplefilter('ignore', np.RankWarning)

# ~~~~~~~~~~~~~~~ Global Parameters ~~~~~~~~~~~~~~~
Tau = 1.; Pr = 1.; Lx = np.pi; Ra_s = .0;
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~ Gap widths l=10~~~~~~~~~~~~~~~
#d = 0.353; Ra_HB_c = 2967.37736364 ; Ra_SS_c = 9853.50008503;

# ~~~~~~~~~~~~~~~ Gap widths l=2~~~~~~~~~~~~~~~
d  = 2.0; Ra = 7.267365e+03 + 10.; 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Self-made packages
from Matrix_Operators import cheb_radial

def Base_State_Coeffs(d):
	"""
	Compute the base state coefficients i.e.

	T_0  = - A_T/r + B_T;

	T'_0 =   A_T/r^2;
	
	"""

	R_1 = 1./d; 
	R_2 = (1. + d)/d;

	A_T = (R_1*R_2)/(R_1 - R_2);
	B_T =      R_1 /(R_1 - R_2)
	
	#D,R = cheb_radial(20,d); 
	#plt.plot(R,-A_T/R + B_T*np.ones(len(R)),'k-');
	#plt.plot(R,A_T/(R*R),'b:');
	#plt.xlim([R[0],R[-1]])
	#plt.show()

	return A_T,B_T;

#@njit(fastmath=True)
def Nusselt(T_hat, R,D,N_fm,nr):

	"""
	Compute the convective Nusselt number:

	Nu−1 = ( ∫ ∂_r T(r,θ) sinθ dθ )/( ∫ ∂_r T_0(r) sinθ dθ)

	"""

	# Compute coefficients
	A_T    = Base_State_Coeffs(d)[0];
	NuM1   = np.zeros(len(R)); 

	# Take every-second component
	for k in range(0,N_fm,2):
		TT       = np.zeros(len(R));
		TT[1:-1] = T_hat[k*nr:(k+1)*nr]; 
		NuM1    += (D@TT)/(1.-(k**2));

	NuM1 = ((R**2)/A_T)*(NuM1/N_fm); # Scale by 1/N due to dct ?

	print("|Nu(R_1) - Nu(R_2)|/|Nu(R_1)| = ",(abs(NuM1[-1] - NuM1[0])/abs(NuM1[0]) ) )
	print("Inner Nu= %e, Outer Nu=%e"%(NuM1[0],NuM1[-1]),"\n")
	return NuM1[0];

def Kinetic_Enegery(X_hat, R,D,N_fm,nr, symmetric = True):

	"""

	Compute the volume integrated kinetic energy


	"""

	from Matrix_Operators import J_theta_RT
	from scipy.fft import dct, idct, dst, idst

	N = N_fm*nr

	IR = np.diag(1./R[1:-1]);
	IR = np.ascontiguousarray(IR); 
	IR2 = IR@IR;

	#'''
	Jψ_hat  = J_theta_RT(X_hat[0:N], nr,N_fm, symmetric)
	Jψ_hat  = np.ascontiguousarray(Jψ_hat);
	Dr      = np.ascontiguousarray(D[1:-1,1:-1]);

	sp = (nr, int( 3*(N_fm/2)) ); # ZERO-PADDED FOR DE-ALIASING !!!!!
	
	u_r_hat = np.zeros(sp); 
	u_θ_hat = np.zeros(sp); 
	
	if symmetric == True:
		for ii in range(1,N_fm,2):
			
			ind_p = ii*nr; 
			ψ_hat = X_hat[ind_p:ind_p+nr];

			u_r_hat[:,ii]  = Jψ_hat[ind_p:ind_p+nr]; #IR2.dot(Jψ_hat[ind_p:ind_p+nr]);
			u_θ_hat[:,ii]  = Dr.dot(ψ_hat); #-IR.dot(Dr.dot(ψ_hat));

	elif symmetric == False:

		for ii in range(N_fm):
			
			ind_p = ii*nr; 
			ψ_hat = X_hat[ind_p:ind_p+nr];

			u_r_hat[:,ii]  = Jψ_hat[ind_p:ind_p+nr]; #IR2.dot(Jψ_hat[ind_p:ind_p+nr]);
			u_θ_hat[:,ii]  = Dr.dot(ψ_hat); #-IR.dot(Dr.dot(ψ_hat));

	u_r = idct( u_r_hat,type=2,axis=-1,norm='ortho',overwrite_x=True);
	u_θ = idst( u_θ_hat,type=2,axis=-1,norm='ortho',overwrite_x=True);
	#'''	

	u_r = IR2@u_r;
	u_θ = -IR@u_θ;

	KE_rθ = abs(u_r)**2 + abs(u_θ)**2;

	θ = np.zeros(N_fm);
	for n in range(N_fm):
		θ[n] = np.pi*( (2*n+1.0)/(2.0*N_fm) );
	norm_coeff = 1./dst(np.sin(θ),type=2,norm='ortho')[0];
	
	#
	# Test radial
	#R_1,R_2 =R[0],R[-1];
	#f   = R**2 - (R_1+R_2)*R + R_1*R_2;
	#f_num = np.linalg.solve(D[0:-1,0:-1],f[0:-1])[0]
	#print("Numerical = ",f_num)
	#f_anal_1 = (R_1**3)/3. - (R_1 + R_2)*0.5*(R_1**2) + (R_1*R_2)*R_1 
	#f_anal_2 = (R_2**3)/3. - (R_1 + R_2)*0.5*(R_2**2) + (R_1*R_2)*R_2 
	#print("Anal = ",f_anal_1 - f_anal_2)

	'''
	# Test radial & theta
	R_1,R_2 =R[0],R[-1];
	f_r = R**2 - (R_1+R_2)*R + R_1*R_2;
	f_θ = np.sin(θ);
	KE_rθ = np.outer(f_r[1:-1],f_θ)
	'''
	# Integrate in θ and take zero mode essentially the IP with 

	# KE = int_r1^r2 (1/2)int_0^π KE(r,θ) r^2 sin(θ) dr dθ
	KE_r       = np.zeros(len(R));
	KE_r[1:-1] = norm_coeff*dst(KE_rθ,type=2,axis=-1,norm='ortho',overwrite_x=True)[:,0];

	# Multiply by r^2 and integrate w.r.t r
	KE_r = R*R*KE_r;
	KE   = np.linalg.solve(D[0:-1,0:-1],KE_r[0:-1])[0];

	V = 2.*(R[-1] - R[0]); # here we divide by 2 as thats what the volume integral gives

	return (1./V)*KE;

def Eq_SYM(X1,R):

	"""
	Function which produces an array of ones and zeros
	such that X_sym*X1 is equatorially symmetric.

	Inputs:
	R  - numpy array radial collocation pts
	X1 - numpy array

	Returns:
	X_sym - numpy array
	
	"""

	X_sym = np.ones(X1.shape);
	nr 	  = len(R[1:-1]);
	N_fm  = int(X1.shape[0]/(3*nr));
	N     = N_fm*nr;

	for k in range(N_fm):

		if k%2 == 0: # eliminate 1,3,5,7, ... (k=0 contains sin(1*\theta), k=1, sin(2*\theta) etc)
			# ψ
			ind = k*nr;
			X_sym[ind:ind+nr] = 0.;

		if k%2 == 1: # eliminate 1,3,5,7, ...
			# T
			ind = N + k*nr;
			X_sym[ind:ind+nr] = 0.;
			
			# C
			ind = 2*N + k*nr;
			X_sym[ind:ind+nr] = 0.;

	return X_sym;

def Build_Matrix_Operators(N_fm,N_r,d):

	"""
	Builds all the matrix operators that have a radial dependence
	and returns these as a tuple of positional arguments

	Inputs:

	N_fm - (int) number of latitudinal modes
	N_r  - (int) number of radial collocation points
	d 	 - (float) gap width

	"""
	
	from Matrix_Operators import R2, kGR_RT
	from scipy.fft import dct, idct, dst, idst


	D,R = cheb_radial(N_r,d); 
	nr  = len(R[1:-1]);

	print("Building Matrices .... \n")
	# Memory Intensive at start-up so replace	

	# 1) Build wave-number weights
	# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~
	# Use correct grid for DCT-II
	θ = np.zeros(N_fm);
	for n in range(N_fm):
		θ[n] = Lx*( (2*n+1.0)/(2.0*N_fm) );	# x in [0,L] 
		
	# Normalization Weights, must be inverses
	akc = np.zeros(N_fm); 
	aks = np.zeros(N_fm);
	for k in range(N_fm):
		ks = k + 1;
		kc = k; 
		akc[k] = 1.0/dct(np.cos(kc*θ),type=2,norm='ortho')[k]; # cosine
		aks[k] = 1.0/dst(np.sin(ks*θ),type=2,norm='ortho')[k]; # sine

	
	#print("f=",f,"\n")
	#print("aks=",aks,"\n")
	#sys.exit()	
	# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~	

	Rsq   =       R2(R,N_fm); 
	gr_K  = kGR_RT(R,N_fm,d); 

	# For T0_J_Theta, as a vector!!!
	A_T = Base_State_Coeffs(d)[0];
	DT0 = A_T/(R[1:-1]**2);

	return D,R,	Rsq,DT0,gr_K,	aks,akc;

def Build_Matrix_Operators_TimeStep(N_fm,N_r,d):

	""""
	Build the non-control parameter dependent matrices to
	perform time-stepping

	Inputs:
	N_fm - (int) number of latitudinal modes
	N_r  - (int) number of radial collocation points
	d 	 - (float) gap width

	"""

	from Matrix_Operators import Nabla2, Nabla4

	D,R,	Rsq,DT0,gr_K,	aks,akc = Build_Matrix_Operators(N_fm,N_r,d);

	# NAB2_BSub_TSTEP
	Nab2 = Nabla2(D,R); Nab2 = np.ascontiguousarray(Nab2);
	R2   = np.diag(R[1:-1]**2); R2 = np.ascontiguousarray(R2);
	I    = np.eye(len(R[1:-1])); I = np.ascontiguousarray(I);
	# ~~~~~~~~~~~~~~~~~~~~~~

	# A4_BSub_TSTEP
	IR   = np.diag(1.0/R); 
	IR2	 = IR@IR;
	D_sq = D@D; 
	
	D4   = Nabla4(D,R);
	D2   = np.matmul(IR2 ,2.0*D_sq - 4.0*(IR@D) + 6.0*IR2 )[1:-1,1:-1];
	A2   = D_sq[1:-1,1:-1];  
	IR2  = IR2[1:-1,1:-1];
	IR4  = IR2@IR2;
	# ~~~~~~~~~~~~~~~~~~~~~~

	nr   = len(R[1:-1]);

	# Make all arrays C continguous where possible

	D4 = np.ascontiguousarray(D4);
	IR4= np.ascontiguousarray(IR4);
	D2 = np.ascontiguousarray(D2);
	A2 = np.ascontiguousarray(A2);
	IR2= np.ascontiguousarray(IR2);
	D  = np.ascontiguousarray(D);
	inv_D = np.linalg.inv(D[0:-1,0:-1]);

	args_Nab2 = (Nab2,R2,I,N_fm,nr);
	args_A4	  = (D4  ,IR4,D2,A2,IR2, N_fm,nr);
	args_FXS  = (inv_D, D,R,N_fm,nr,aks,akc);

	return D,R,	Rsq,DT0,gr_K, N_fm,nr, args_Nab2,args_A4,args_FXS;


def _Newton(X,Ra, N_fm,N_r,d, dt = 10**4, tol_newton = 1e-10,tol_gmres = 1e-04,Krylov_Space_Size = 150, symmetric = True):
	
	"""
	Given a starting point and a control parameter compute a new steady-state using Newton iteration

	Inputs:

	X  - starting guess (numpy vector)
	Ra - new parameter (float)
	args_f - tuple of arguments to be supplied to solver

	defaults
	tol_newton - (float) newton iteration tolerance to terminate iterations
	tol_gmres  - (float) GMRES x =A^-1 b solver algorithim parameter
	Krylov_Space_Size - (int) number of Krylov vectors to use when forming the subspace

	Returns:
	X_new  - new solution (numpy vector) *if converged*

	"""

	from Matrix_Operators import NLIN_DFX as DFX
	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	
	D,R,	Rsq,DT0,gr_k,N_fm,nr,	args_Nab2,args_A4,args_FX = Build_Matrix_Operators_TimeStep(N_fm,N_r,d);

	'''
	args_Nab2_T = args_Nab2; args_Nab2_S = args_Nab2;
	from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP
	'''
	from Matrix_Operators import A4_BSub_TSTEP_V2   as   A4_BSub_TSTEP
	from Matrix_Operators import NAB2_BSub_TSTEP_V2 as NAB2_BSub_TSTEP
	
	from Matrix_Operators import   A4_TSTEP_MATS
	from Matrix_Operators import NAB2_TSTEP_MATS

	L4_inv   =   A4_TSTEP_MATS( Pr*dt,N_fm,nr,D,R); args_A4     = (L4_inv,D,R,N_fm,nr);
	L2_inv_T = NAB2_TSTEP_MATS(    dt,N_fm,nr,D,R); args_Nab2_T = (L2_inv_T,N_fm,nr);
	L2_inv_S = NAB2_TSTEP_MATS(Tau*dt,N_fm,nr,D,R); args_Nab2_S = (L2_inv_S,N_fm,nr);
	#'''

	#from Matrix_Operators import kGR
	#Gr = kGR(R,N_fm,d)


	import scipy.sparse.linalg as spla # Used for Gmres Netwon Sovler

	nr = int( X.shape[0]/(3*N_fm) );
	N  = N_fm*nr; 
	dv = np.random.randn(X.shape[0]);
	
	error     = 1.0; 
	iteration = 0; 
	exit      = 1.0; 

	if symmetric == True:
		X_SYM  = Eq_SYM(X,R);
		X 	   = X_SYM*X;
	else:
		X_SYM  = np.ones(X.shape);

	def PFX(Xn):

		NX 	= np.zeros(3*N);
		
		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# Linear
		NX[:] = -1.*dt*FX(Xn, *args_FX,     symmetric); # 36% FIX
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		NX[0:N]     += Ω + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        =   A4_BSub_TSTEP(NX[0:N],     *args_A4,     Pr*dt, symmetric) - ψ;

		# 2) Temperature - T
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2_T,    dt, symmetric) - T;

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2_S,Tau*dt, symmetric) - S;

		return np.hstack((ψ_new,T_new,S_new));

	def PDFX(dv,Xn):

		NX 	   = np.zeros(3*N);

		δψ = dv[0:N];
		δT = dv[N:2*N];
		δS = dv[2*N:3*N];

		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# Linear
		NX[:]  = -1.*dt*DFX(dv,Xn, *args_FX,     symmetric); # 36% FIX
		δψ_T0  = DT0_theta(δψ,   	DT0,N_fm,nr, symmetric);
		δΩ     = A2_SINE(δψ,     	D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - ∆Ω
		NX[0:N]     += δΩ + dt*Pr*gr_k.dot(Ra*δT - Ra_s*δS);
		ψ_new        =   A4_BSub_TSTEP(NX[0:N],     *args_A4,     Pr*dt, symmetric) - δψ;

		# 2) Temperature - ∆T
		NX[N:2*N]   += Rsq.dot(δT) - dt*δψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2_T,    dt, symmetric) - δT;

		# 3) Solute - ∆S
		NX[2*N:3*N] += Rsq.dot(δS) - dt*δψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2_S,Tau*dt, symmetric) - δS;

		return np.hstack((ψ_new,T_new,S_new));	

	while error > tol_newton:
		
		X = X_SYM*X;	
		
		fx  = PFX(X);
		DF  = lambda dv: PDFX(dv,X);
		
		dfx = spla.LinearOperator((X.shape[0],X.shape[0]),matvec=DF,dtype='float64');

		# Solve pre-conditioned DF(X)*dv = F(X); 
		b_norm 	= np.linalg.norm(fx,2);
		dv,exit = spla.lgmres(dfx,fx, tol = tol_gmres*b_norm, maxiter=250, inner_m = Krylov_Space_Size);
		X 		= X - dv;
		error 	= np.linalg.norm(dv,2)/np.linalg.norm(X,2);

		print('Iteration = %d, Error = %e'%(iteration,error),"\n")
		iteration+=1

	# Compute diagnoistics	
	KE = Kinetic_Enegery(X[0:N], R,D,N_fm,nr, symmetric)
	NuT= Nusselt(X[N:2*N]  ,R,D,N_fm,nr);
	NuS= Nusselt(X[2*N:3*N],R,D,N_fm,nr);

	return X,KE,NuT,NuS;

def Newton(filename='blah',frame=-1):

	"""
	Given an initial condition and full parameter specification time-step the system
	using time-stepping scheme CNAB1

	Inputs:

	filename
	defaults
	dt - (float) default time-step for time-integration

	Returns:

	"""

	# l =2 mode
	N_fm = 20; 
	N_r  = 20;
	d    = 2.0; Ra = 7.267365e+03 + 10.

	# ~~~~~~~~~ Random Initial Conditions ~~~~~~~~~~~~~~~~
	D,R  = cheb_radial(N_r,d); 
	nr   = len(R[1:-1]);
	N 	 = nr*N_fm;

	X = np.random.rand(3*N);
	X = 1e-03*(X/np.linalg.norm(X,2))

	#Ra = Ra_HB_c + 2.;
	start_time = 0.;
	# ~~~~~~~~~ Old Initial Conditions ~~~~~~~~~~~~~~~~~~
	if filename.endswith('.npy'):
		Y  = np.load("Y_Nt300_Nr30_l10_Ra100.npy");
		X  = Y[0:-1]; 
		Ra = Y[-1,0];

	# ~~~~~~~~~ New Initial Conditions ~~~~~~~~~~~~~~~~~~
	if filename.endswith('.h5'):

		f = h5py.File(filename, 'r+')

		# Problem Params
		X      = f['Checkpoints/X_DATA'][frame];
		Ra     = f['Parameters']["Ra"][()];
		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]
		d 	   = f['Parameters']["d"][()]
		#start_time = f['Parameters']["start_time"][()];
		try:
			start_time  = f['Scalar_Data/Time'][()][frame]
		except:
			pass;
		f.close();

		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(start_time,Ra,d,N_fm,N_r))    

		
		# ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		fac_R =1; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		fac_T =1; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Save all data in the same structure as time-stepping	
	X_DATA  = [];
	KE  = []; 
	NuT = []; 
	NuS = [];

	X_new,ke,nuS,nuT = _Newton(X,Ra, N_fm,N_r,d, Krylov_Space_Size = 150, symmetric = False)
		
	KE.append( ke );
	NuT.append( nuT );
	NuS.append( nuS );
	X_DATA.append(X_new);

	# Save the different errors 
	Nstep_file = h5py.File('Newton_Iteration_Data.h5', 'w')

	# Checkpoints
	Checkpoints = Nstep_file.create_group("Checkpoints");
	Checkpoints['X_DATA'] = X_DATA;

	# Scalar Data
	Scalar_Data = Nstep_file.create_group("Scalar_Data")
	Scalar_Data['KE']   = KE;
	Scalar_Data['Nu_T'] = NuT;
	Scalar_Data['Nu_S'] = NuS;
	Scalar_Data['Time'] = [];

	# Problem Params
	Parameters = Nstep_file.create_group("Parameters");
	for key,val in {"Ra":Ra,"d":d,	"N_r":N_r,"N_fm":N_fm,"dt":10**4,	"start_time":0.}.items():
		Parameters[key] = val;

	Nstep_file.close();  

	return None;


def _Time_Step(X,Ra, N_fm,N_r,d, start_time = 0., Total_time = 1./Tau, dt=1e-04, symmetric = True):


	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	
	D,R,	Rsq,DT0,gr_k,N_fm,nr,	args_Nab2,args_A4,args_FX = Build_Matrix_Operators_TimeStep(N_fm,N_r,d);

	'''
	args_Nab2_T = args_Nab2; args_Nab2_S = args_Nab2;
	from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP
	'''
	from Matrix_Operators import A4_BSub_TSTEP_V2   as   A4_BSub_TSTEP
	from Matrix_Operators import NAB2_BSub_TSTEP_V2 as NAB2_BSub_TSTEP
	
	from Matrix_Operators import   A4_TSTEP_MATS
	from Matrix_Operators import NAB2_TSTEP_MATS

	L4_inv   =   A4_TSTEP_MATS( Pr*dt,N_fm,nr,D,R); args_A4     = (L4_inv,D,R,N_fm,nr);
	L2_inv_T = NAB2_TSTEP_MATS(    dt,N_fm,nr,D,R); args_Nab2_T = (L2_inv_T,N_fm,nr);
	L2_inv_S = NAB2_TSTEP_MATS(Tau*dt,N_fm,nr,D,R); args_Nab2_S = (L2_inv_S,N_fm,nr);
	#'''

	Time    = []; 
	X_DATA  = [];
	KE  = [];  
	NuT = []; 
	NuS = [];

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	if symmetric == True:
		X_SYM  = Eq_SYM(X,R);
		X 	   = X_SYM*X;
	else:
		X_SYM  = np.ones(X.shape);

	error     = 1.0;
	iteration = 0;
	N_iters   = int(Total_time/dt);
	N_save    = N_iters/10;
	N_print   = 100;
	N         = int(X.shape[0]/3)

	#from Matrix_Operators import kGR
	#Gr = kGR(R,N_fm,d)

	#@profile
	def Step_Python(Xn,kinetic=False):

		NX= np.zeros(3*N);

		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		#'''
		if kinetic == True:
			OUT 	= 		 FX(Xn, *args_FX,     symmetric,kinetic); # 36% FIX
			NX[:],KE= -1.*dt*OUT[0],OUT[1];
		else:
			NX[:]	= -1.*dt*FX(Xn, *args_FX,     symmetric,kinetic); # 36% FIX
		#'''
		#KE = Kinetic_Enegery(Xn, R,D,N_fm,nr, symmetric);
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,   D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		NX[0:N]     += Ω + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        =   A4_BSub_TSTEP(NX[0:N],     *args_A4,     Pr*dt, symmetric);

		# 2) Temperature - T
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2_T,    dt, symmetric);

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2_S,Tau*dt, symmetric);

		if kinetic == True:
			return np.hstack((ψ_new,T_new,S_new)),KE;
		else:
			return np.hstack((ψ_new,T_new,S_new));

	timer = 0.	
	while iteration < N_iters:

		ST_time = time.time()
		X_new,kinetic = Step_Python(X,True);
		END_time = time.time();
		
		#KE = Kinetic_Enegery(Xn, R,D,N_fm,nr, symmetric);
		#val = Kinetic_Enegery(X, R,D,N_fm,nr, symmetric = False); print(val)	

		#NuT.append( np.linalg.norm(X_new,2) )
		#NuS.append( np.linalg.norm(X_new[2*N:3*N],2) )

		KE.append(  abs(kinetic) );
		NuT.append( Nusselt(X_new[N:2*N]  ,R,D,N_fm,nr) );
		NuS.append( Nusselt(X_new[2*N:3*N],R,D,N_fm,nr) );
		Time.append(start_time + dt*iteration);

		if iteration%N_print == 0:
			
			print("Iteration =%i, Time = %e, Energy = %e \n "%(iteration,start_time + dt*iteration,KE[-1]))

		if iteration%N_save == 0:	


			print("Saving full solution vector X \n");
			X_DATA.append(X_new);

			# Save the different errors 
			Tstep_file = h5py.File('Time_Integration_Data.h5', 'w')

			# Checkpoints
			Checkpoints = Tstep_file.create_group("Checkpoints");
			Checkpoints['X_DATA'] = X_DATA;

			# Scalar Data
			Scalar_Data = Tstep_file.create_group("Scalar_Data")
			Scalar_Data['KE']   = KE;
			Scalar_Data['Nu_T'] = NuT;
			Scalar_Data['Nu_S'] = NuS;
			Scalar_Data['Time'] = Time;

			# Problem Params
			Parameters = Tstep_file.create_group("Parameters");
			for key,val in {"Ra":Ra,"d":d,	"N_r":N_r,"N_fm":N_fm,"dt":dt,	"start_time":start_time}.items():
				Parameters[key] = val;

			Tstep_file.close();  		

		X = X_SYM*X_new;	
		iteration+=1;    
		if iteration > 1:
			timer += END_time - ST_time;
	print("Avg time = %e \n"%(timer/(iteration-1)) );

	return X_new;

def Time_Step(filename='blah',frame=-1):

	"""
	Given an initial condition and full parameter specification time-step the system
	using time-stepping scheme CNAB1

	Inputs:

	filename
	defaults
	dt - (float) default time-step for time-integration

	Returns:

	"""

	# l=10 mode
	#N_fm = 2*10; 
	#N_r  = 20;
	#d    = 0.353; Ra = 2853.5 + 1. 

	# l =2 mode
	N_fm = 20; 
	N_r  = 20;
	#d    = 2.0; Ra = 7.267365e+03 + 10.
	d    = 2.0; Ra = 6.77*(10**3) + 1.

	# ~~~~~~~~~ Random Initial Conditions ~~~~~~~~~~~~~~~~
	D,R  = cheb_radial(N_r,d); 
	nr   = len(R[1:-1]);
	N 	 = nr*N_fm;

	X = np.random.rand(3*N);
	X = (X/np.linalg.norm(X,2))
	
	start_time = 0.;
	# ~~~~~~~~~ Old Initial Conditions ~~~~~~~~~~~~~~~~~~
	if filename.endswith('.npy'):
		Y  = np.load("Y_Nt300_Nr30_l10_Ra100.npy");
		X  = Y[0:-1]; 
		Ra = Y[-1,0];

	# ~~~~~~~~~ New Initial Conditions ~~~~~~~~~~~~~~~~~~
	if filename.endswith('.h5'):

		f = h5py.File(filename, 'r+')

		# Problem Params
		X      = f['Checkpoints/X_DATA'][frame];
		Ra     = f['Parameters']["Ra"][()];
		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]
		d 	   = f['Parameters']["d"][()]
		try:
			start_time  = f['Scalar_Data/Time'][()][frame]
		except:
			pass;
		f.close();    

		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(start_time,Ra,d,N_fm,N_r))    

		
		# ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		fac_R =1; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		fac_T =1; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Total_time = 5*(10**3)*(1./Tau);

	X_new = _Time_Step(X,Ra, N_fm,N_r,d, start_time, Total_time, dt=5e-02, symmetric = False);

	return None;


# Execute main
if __name__ == "__main__":
	
	file = 'Time_Integration_Data.h5'; frame = -1;
	
	Newton(file,frame);
	
	#file ='Newton_Iteration_Data.h5'; frame = 0;
	#Time_Step();#(file,frame);