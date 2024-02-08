from numba import njit
import numpy as np
import matplotlib.pyplot as plt

import os, time, warnings, h5py
warnings.simplefilter('ignore', np.RankWarning)

# File handling methods
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 0

    while os.path.exists(path):
        path = filename.split('_')[0] + "_" +str(counter) + extension
        counter += 1

    return path

def print_h5py(filename):

	f = h5py.File(filename,'r')	

	print('\n')
	print('#~~~~~~~ Parameters ~~~~~~~~~~~~~~')
	for key,value in f['Parameters'].items():
		print(key,'=',value[()])
	print('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
	
	f.close()

	return None;

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

@njit(fastmath=True)
def Nusselt(T_hat, d, R,D,N_fm,nr):

	"""
	Compute the convective Nusselt number:

	Nu−1 = ( ∫ ∂_r T(r,θ) sinθ dθ )/( ∫ ∂_r T_0(r) sinθ dθ) for r=R2 or R1

	comes from volume integrating the eqns w.r.t bcss 
	"""

	# Compute coefficients
	R_1 = 1./d; 
	R_2 = (1. + d)/d;
	A_T = (R_1*R_2)/(R_1 - R_2);

	NuM1 = 0.*R
	for k in range(0,N_fm,2):
		TT      = 0.*R;
		TT[1:-1]= T_hat[k*nr:(k+1)*nr]
		NuM1   += (D@TT)/(1.-(k**2));
	NuM1 = ((R**2)/A_T)*NuM1;

	# error = abs(NuM1[-1] - NuM1[0])/abs(NuM1[-1]);	
	# if error > 1e-03:
	# 	print("|Nu(R_1) - Nu(R_2)|/|Nu(R_1)| = ",error," > .1%, Increase Nr \n" )
	
	return NuM1[0];

def Kinetic_Energy(X_hat, R,D,N_fm,nr):
	"""
	Compute the volume integrated kinetic energy

	KE = (1/2)*(1/V) int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ

	where

	V = int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ = (2/3)*(r2^3 - r1^3)
	"""
	from Matrix_Operators import J_theta_RT
	from Transforms import IDCT,IDST,grid

	N   = N_fm*nr;
	Dr  = np.ascontiguousarray(D[1:-1,1:-1]);

	JPSI = J_theta_RT(X_hat[0:N], nr,N_fm)#.reshape((nr,N_fm)); # ~ Cosine

	sp = (nr, N_fm);
	JT_psi_hat = np.zeros(sp); 	
	Dpsi_hat   = np.zeros(sp); 

	# O(nr^2*N_fm)
	for ii in range(N_fm):

		ind_p = ii*nr; 
		psi   = X_hat[ind_p:ind_p+nr];

		Dpsi_hat[:,ii]    = Dr.dot(psi); # Sine					
		JT_psi_hat[:,ii]  = JPSI[ind_p:ind_p+nr];  # Cosine
	
	# Convert Sine to sinusoids
	Dpsi_hat[:,1:] = Dpsi_hat[:,0:-1];  Dpsi_hat[:,0] = 0.0;

	Jψ    = IDCT(JT_psi_hat,n = (3*N_fm)//2) 
	dr_ψ  = IDST(Dpsi_hat  ,n = (3*N_fm)//2)

	IR2   = np.diag(1./(R[1:-1]**2));
	IR2   = np.ascontiguousarray(IR2);

	θ     = grid((3*N_fm)//2)
	
	KE_rθ = (IR2@Jψ)**2  +  dr_ψ**2;
	KE_θ  = np.trapz(KE_rθ         ,x=R[1:-1],axis=0 )
	KE    = np.trapz(KE_θ*np.sin(θ),x=θ      ,axis=-1)
	V     = (2./3.)*(R[-1]**3 - R[0]**3);

	return (.5/V)*KE;

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

		if k%2 == 0: 

			# eliminate modes 1,3,5,7, ... (k=0 contains sin(1*\theta), k=1, sin(2*\theta) etc)
			# ψ
			ind = k*nr;
			X_sym[ind:ind+nr] = 0.;
		
		if k%2 == 1: 	
			# eliminate 1,3,5,7, ...
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
	
	D,R = cheb_radial(N_r,d); 
	nr  = len(R[1:-1]);

	Rsq   =       R2(R,N_fm); 
	gr_K  = kGR_RT(R,N_fm,d); 

	# For T0_J_Theta, as a vector!!!
	A_T = Base_State_Coeffs(d)[0];
	DT0 = A_T/(R[1:-1]**2);

	return D,R,	Rsq,DT0,gr_K;

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

	D,R,	Rsq,DT0,gr_K = Build_Matrix_Operators(N_fm,N_r,d);

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

	args_Nab2 = (Nab2,R2,I,N_fm,nr);
	args_A4	  = (D4  ,IR4,D2,A2,IR2, N_fm,nr);
	args_FXS  = (D,R,N_fm,nr);

	return D,R,	Rsq,DT0,gr_K, N_fm,nr, args_Nab2,args_A4,args_FXS;


def _Time_Step(X,Ra,Ra_s,Tau,Pr,d,	N_fm,N_r, save_filename = 'TimeStep_0.h5', start_time = 0., Total_time = 1., dt=1e-04, symmetric = True, linear=False, Verbose=False):

	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP

	D,R,	Rsq,DT0,gr_k,N_fm,nr,	args_Nab2,args_A4,args_FX = Build_Matrix_Operators_TimeStep(N_fm,N_r,d);
	
	Time    = []; 
	X_DATA  = [];
	KE  = [];  
	NuT = []; 
	NuS = [];
	Norm = [];

	#~~~~~~~~~~~~~~~~~~~~~~~~~~
	if symmetric == True:
		X_SYM  = Eq_SYM(X,R);
		X 	   = X_SYM*X;
	else:
		X_SYM  = np.ones(X.shape);

	X_SYM     = Eq_SYM(X,R);	
	error     = 1.0;
	iteration = 0;
	N_iters   = int(Total_time/dt);
	N_save    = int(N_iters/10);
	N_print   = 10;
	N         = int(X.shape[0]/3)

	def Step_Python(Xn,linear):

		NX= np.zeros(3*N);

		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		if linear == False:
			OUT 	=  FX(Xn, *args_FX, symmetric); # 36% FIX
			NX[:],KE= -1.*dt*OUT[0],OUT[1];
		else:
			KE = Kinetic_Energy(Xn, R,D,N_fm,nr);
		
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		# Here we invert the LHS = ( \hat{A}^2 − ∆t Pr \hat{A}^4)
		NX[0:N]     +=  Ω   + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        =  A4_BSub_TSTEP(NX[0:N],     *args_A4,     Pr*dt, symmetric); 

		# 2) Temperature - T
		# Here we invert the LHS = r^2( I − ∆t \hat{\nabla}^2)
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2,    dt, symmetric);

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2,Tau*dt, symmetric);  
	
		return np.hstack((ψ_new,T_new,S_new)),KE;

	timer = 0.	
	while iteration < N_iters:

		ST_time = time.time()
		X_new,kinetic = Step_Python(X,linear);
		END_time = time.time();
		
		Norm.append( np.linalg.norm(X_new,2) );
		KE.append(  abs(kinetic) ); #
		NuT.append( Nusselt(X_new[N:2*N]  ,d,R,D,N_fm,nr) );
		NuS.append( Nusselt(X_new[2*N:3*N],d,R,D,N_fm,nr) );
		Time.append(start_time + dt*iteration);

		if (iteration%N_print == 0) and (Verbose == True):
			print("Iteration =%i, Time = %e, Energy = %e, NuT = %e \n "%(iteration,start_time + dt*iteration,KE[-1],NuT[-1]))

		if iteration%N_save == 0:	

			X_DATA.append(X_new);

			f = h5py.File(save_filename,'w')

			Checkpoints = f.create_group("Checkpoints");
			Checkpoints['X_DATA'] = X_DATA;

			Scalar_Data = f.create_group("Scalar_Data")
			Scalar_Data['Norm'] = Norm;
			Scalar_Data['KE']   = KE;
			Scalar_Data['Nu_T'] = NuT;
			Scalar_Data['Nu_S'] = NuS;
			Scalar_Data['Time'] = Time;

			Parameters = f.create_group("Parameters");
			for key,val in {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,	"N_r":N_r,"N_fm":N_fm,"dt":dt,	"start_time":start_time, "symmetric":symmetric}.items():
				Parameters[key] = val;

			f.close();  		

		X = X_SYM*X_new;	
		iteration+=1;    
		if iteration > 1:
			timer += END_time - ST_time;
	print("Avg time = %e \n"%(timer/(iteration-1)) );

	return X_new;

def Time_Step(open_filename='TimeStep_0.h5',save_filename = 'TimeStep_0.h5',frame=-1):

	"""
	Given an initial condition and full parameter specification time-step the system
	using time-stepping scheme CNAB1

	Inputs:

	Returns:

	"""

	start_time = 0.;
	Total_time = .5;

	# ~~~~~~~~~~~~~~~ Gap widths l=10~~~~~~~~~~~~~~~
	Tau = 1/15; Ra_s = 500; Pr=1;
	d = 0.353;  Ra   = 9853.50008503 - 0.01;

	N_fm = 2**7;
	N_r  = 2**5;

	# ~~~~~~~~~ Random Initial Conditions ~~~~~~~~~~~~~~~~
	N = (N_r - 1)*N_fm;
	X = np.random.rand(3*N);
	X = 1e-03*(X/np.linalg.norm(X,2))
	
	# ~~~~~~~~~ Initial Conditions ~~~~~~~~~~~~~~~~~~
	if open_filename.endswith('.h5'):

		f = h5py.File(open_filename, 'r+')

		# Problem Params
		X      = f['Checkpoints/X_DATA'][frame];
		
		Ra     = f['Parameters']["Ra"][()];
		Ra_s   = f['Parameters']["Ra_s"][()];
		Tau    = f['Parameters']["Tau"][()];
		Pr     = f['Parameters']["Pr"][()];
		d 	   = f['Parameters']["d"][()]

		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]

		st_time= f['Scalar_Data/Time'][()][frame]
		f.close();    

		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))    

		# ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		fac_R =1; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		fac_T =1; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	filename = uniquify(save_filename)
	kwargs  = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
	X_new   = _Time_Step(X,**kwargs, save_filename = filename ,dt=1e-03, symmetric=True,linear=True,Verbose=True);

	return None;


def _Newton(X,	Ra,Ra_s,Tau,Pr,d,	N_fm,N_r,symmetric = True, dt=10**4,	tol_newton = 1e-10,tol_gmres = 1e-04,Krylov_Space_Size = 150):
	
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

	import scipy.sparse.linalg as spla # Used for Gmres Netwon Sovler

	from Matrix_Operators import NLIN_DFX as DFX
	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP

	D,R,	Rsq,DT0,gr_k,N_fm,nr,	args_Nab2,args_A4,args_FX = Build_Matrix_Operators_TimeStep(N_fm,N_r,d);
	
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
		
		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# Linear
		NX    = -1.*dt*FX(Xn, *args_FX, symmetric)[0]; # 36% FIX
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		NX[0:N]     += Ω + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        =   A4_BSub_TSTEP(NX[0:N],     *args_A4,   Pr*dt, symmetric) - ψ;

		# 2) Temperature - T
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2,    dt, symmetric) - T;

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2,Tau*dt, symmetric) - S;

		return np.hstack((ψ_new,T_new,S_new));

	def PDFX(dv,Xn):

		δψ = dv[0:N];
		δT = dv[N:2*N];
		δS = dv[2*N:3*N];

		# Linear
		NX     = -1.*dt*DFX(dv,Xn, *args_FX,     symmetric); # 36% FIX
		δψ_T0  = DT0_theta(δψ,   	DT0,N_fm,nr, symmetric);
		δΩ     = A2_SINE(δψ,     	D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - ∆Ω
		NX[0:N]     += δΩ + dt*Pr*gr_k.dot(Ra*δT - Ra_s*δS);
		ψ_new        =   A4_BSub_TSTEP(NX[0:N],     *args_A4,   Pr*dt, symmetric) - δψ;

		# 2) Temperature - ∆T
		NX[N:2*N]   += Rsq.dot(δT) - dt*δψ_T0;
		T_new        = NAB2_BSub_TSTEP(NX[N:2*N],   *args_Nab2,    dt, symmetric) - δT;

		# 3) Solute - ∆S
		NX[2*N:3*N] += Rsq.dot(δS) - dt*δψ_T0;
		S_new        = NAB2_BSub_TSTEP(NX[2*N:3*N], *args_Nab2,Tau*dt, symmetric) - δS;

		return np.hstack((ψ_new,T_new,S_new));	

	while (error > tol_newton) and (iteration < 20):
		
		X = X_SYM*X;	
		
		fx  = PFX(X);
		DF  = lambda dv: PDFX(dv,X);
		
		dfx = spla.LinearOperator((X.shape[0],X.shape[0]),matvec=DF,dtype='float64');

		# Solve pre-conditioned DF(X)*dv = F(X); 
		b_norm 	= np.linalg.norm(fx,2);
		dv,exit = spla.lgmres(dfx,fx, maxiter=250, inner_m = Krylov_Space_Size, atol = tol_gmres*b_norm);
		X 		= X - dv;
		error 	= np.linalg.norm(dv,2)/np.linalg.norm(X,2);

		print('Newton Iteration = %d, Error = %e'%(iteration,error),"\n")
		iteration+=1

	if (iteration == 20) or (exit != 0):

		print('Newton iteration could not converge');
		return _,	_,_,_,_, False;
	
	else:	
		
		# Compute diagnoistics
		Norm = np.linalg.norm(X,2);
		KE   = Kinetic_Energy(X,    R,D,N_fm,nr);
		NuT  = Nusselt(X[N:2*N]  ,d,R,D,N_fm,nr);
		NuS  = Nusselt(X[2*N:3*N],d,R,D,N_fm,nr);

		return X,	Norm,KE,NuT,NuS, True;

def Newton(open_filename='NewtonSolve_0.h5',save_filename = None,frame=-1):

	"""
	Given an initial condition and full parameter specification solve 
	the governing equations using Newton iteration

	Inputs:

	Returns:

	"""

	#~~~~~~~~~~#~~~~~~~~~~#
	# Load data
	#~~~~~~~~~~#~~~~~~~~~~#
	if open_filename.endswith('.h5'):

		f = h5py.File(open_filename, 'r+')

		# Problem Params
		X      = f['Checkpoints/X_DATA'][frame];
		
		Ra     = f['Parameters']["Ra"][()];
		Ra_s   = f['Parameters']["Ra_s"][()];
		Tau    = f['Parameters']["Tau"][()];
		Pr     = f['Parameters']["Pr"][()];
		d 	   = f['Parameters']["d"][()]

		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]

		st_time= f['Scalar_Data/Time'][()][frame]
		f.close();

		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))    

		# # ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		# from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		# fac_R =1; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		# fac_T =1; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	# ~~~~~~~~~ Old Initial Conditions ~~~~~~~~~~~~~~~~~~
	if open_filename.endswith('.npy'):
		X  = np.load(open_filename);

		Ra     = 9853.50008503 - 1e-02
		Ra_s   = 500.
		Tau    = 1./15.;
		Pr     = 1.
		d 	   = 0.353

		N_fm   = 64; #300
		N_r    = 20; #30

		symmetric = True
		st_time   = 0.

		#X /= 0.01*np.linalg.norm(X,2)

	#~~~~~~~~~~#~~~~~~~~~~#
	# Run Code
	#~~~~~~~~~~#~~~~~~~~~~#
	kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
	X_new,Norm,KE,NuT,NuS,_ = _Newton(X,**kwargs,symmetric = True,tol_newton = 1e-8)

	
	#~~~~~~~~~~#~~~~~~~~~~#
	# Save data
	#~~~~~~~~~~#~~~~~~~~~~# 
	if save_filename == None:
		filename = uniquify(open_filename)
	else:
		filename = uniquify(save_filename)
	
	f = h5py.File(filename,'w')

	Checkpoints = f.create_group("Checkpoints");
	Checkpoints['X_DATA'] = [X_new];

	Scalar_Data = f.create_group("Scalar_Data")
	Scalar_Data['Norm'] = [Norm];
	Scalar_Data['KE']   = [KE];
	Scalar_Data['Nu_T'] = [NuT];
	Scalar_Data['Nu_S'] = [NuS];
	Scalar_Data['Time'] = [0.0];

	Parameters = f.create_group("Parameters");
	for key,val in kwargs.items():
		Parameters[key] = val;
	
	f.close();

	from Plot_Tools import Cartesian_Plot
	Cartesian_Plot(filename,frame=-1)

	return None;


'''
def _ContinC(Y_0_dot,Y_0,sign,ds, *args_f):

	"""
	
	

	"""
	from Matrix_Operators import NLIN_DFX as DFX
	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	
	D,R,	Rsq,DT0,gr_k,N_fm,nr,	args_Nab2,args_A4,args_FX = Build_Matrix_Operators_TimeStep(N_fm,N_r,d);

	args_Nab2_T = args_Nab2; 
	args_Nab2_S = args_Nab2;
	from Matrix_Operators import A4_BSub_TSTEP, NAB2_BSub_TSTEP

	import scipy.sparse.linalg as spla # Used for Gmres Netwon Sovler

	
	# Shallow copy as mutable
	X_0  = Y_0[0:-1].copy();
	µ_0  = Y_0[  -1].copy();

	if symmetric == True:
		X_SYM  = Eq_SYM(X_0,R);
		X_0    = X_SYM*X_0;
	
	nr = int( X_0.shape[0]/(3*N_fm) );
	N  = N_fm*nr; 
	dv = np.random.randn(X_0.shape[0]);

	# Hyper-parameter to balance IP
	δ  = 1./(3.*N);
	
	def PFX(Xn,Ra):

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

	def PDFX(dv,Xn,Ra):

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

	def PDFµ(Xn):
		
		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# DF(X,µ)_µ
		dfµ       = np.zeros(Xn.shape[0])
		fµ		  = dt*Pr*gr_k.dot(T)	
		dfµ[0:N]  = A4_BSub_TSTEP( fµ , *args_A4, Pr*dt, symmetric);
	
		return dfµ;


	# 1) ~~~~~~~~~~~~~~~~~ Compute Prediction ~~~~~~~~~~~~~~~~~ 
	if np.linalg.norm(Y_dot,2) == 0:
		
		# P*DF(X,µ)_µ	
		dfµ  = (-1.)*PDFµ(X_0)

		# P*DF(X,µ)_X
		DF_X = lambda dv: PDFX(dv,X_0,µ_0);
		dfx  = spla.LinearOperator((X.shape[0],X.shape[0]),matvec=DF_X,dtype='float64');

		# a) Solve pre-conditioned P*DF_X(X_0)*ξ = (-1)*P*DF_µ(X); 
		b_norm 	= np.linalg.norm(dfµ,2);
		ξ,exit  = spla.lgmres(dfx,dfµ, tol = tol_gmres*b_norm, maxiter=250, inner_m = Krylov_Space_Size);

		if (exit != 0): 
			print("\n Warning: couldn't compute prediction for ξ \n")
			return _,_,_,_,_,False;

		# b) ~~~~~~~~~~~~~~~~~ Compute x_dot, µ_dot algebra ~~~~~~~~~~
		µ_dot = sign/np.sqrt( 1.0 + δ*(np.linalg.norm(ξ,2)-1.0) );
		X_dot = µ_dot*ξ; 
		
		print("mu_dot ",mu_dot);

	else:

		# Shallow copy as mutable
		X_dot = Y_0_dot[0:-1].copy(); 
		µ_dot = Y_0_dot[  -1].copy();	

	# Make prediction	
	X = X_0 + X_dot*ds; 
	µ = µ_0 + µ_dot*ds;
	Y = np.hstack( (X,µ) );
		
	#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~

	G  = np.zeros(Y.shape[0])
	
	err_X,err_µ = 1.0,1.0; 
	iteration   = 0; 
	exit        = 1.0; 

	while ( (err_X > tol_netwon) or (err_µ > tol_netwon) ):
		
		X  = Y[0:-1].copy();
		µ  = Y[  -1].copy();

		# P*DF(X,µ)_X & P*F(X,µ)_µ
		DF_X    = lambda dv: PDFX(dv,X,µ);
		DF_µ    =            PDFµ(X)
		
		# F(X,µ) & p(X,µ,s) 
		G[0:-1] = PFX(X,µ)
		G[  -1] = δ*np.dot(X_dot,X - X_0)  + (1.-δ)*µ_dot*(µ - µ_0) - ds;
		
		# 3) Compute DG_y
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~
		DG  = lambda dY: np.vstack( ( DF_X(          dY[0:-1]) 		   + DF_µ*dY[-1], \
									  δ*np.dot(X_dot,dY[0:-1]) + (1.-δ)*µ_dot*dY[-1] ) );
		
		DGy = spla.LinearOperator((Y.shape[0],Y.shape[0]),matvec=DG,dtype='float64');

		b_norm  = np.sqrt( δ*np.dot(G[0:-1],G[0:-1]) + (1.-δ)*(G[-1]**2) );
		dY,exit = spla.lgmres(DGy,G, tol = tol_gmres*b_norm, maxiter=250, inner_m = Krylov_Space_Size);
		Y 		= Y - dY;

		# 4) ~~~~~~~~~~~~~~~~~ Compute error ~~~~~~~~~~~~~~~~~
		err_X = np.linalg.norm(dY[0:-1],2)/np.linalg.norm(X,2);
		err_µ = abs(dY[-1])/abs(µ);
		
		print('Psuedo-Arc Iteration = %d, Error X = %e, Error µ = %e'%(iteration,err_X,err_µ),"\n") 
		iteration+=1;

		# 5)  ~~~~~~~~~~~~~~~~~  ds control  ~~~~~~~~~~~~~~~~~ 
		if (iteration > 10) or (exit != 0):
			
			iteration=0;
			ds*=0.5;

			X = X_0 + X_dot*ds; 
			µ = µ_0 + µ_dot*ds;	
			Y = np.hstack( (X,µ) );

			if ds < 1e-06:
				print("\n Warning small step-size terminating solve \n")	
				return _,_,_,_,_,False;

		elif (iteration < 5) and (exit == 0):

			ds*=2.;

	# Compute diagnoistics	
	KE = Kinetic_Enegery(Y[0:N], R,D,N_fm,nr, symmetric)
	NuT= Nusselt(Y[N:2*N]  ,d,R,D,N_fm,nr);
	NuS= Nusselt(Y[2*N:3*N],d,R,D,N_fm,nr);		

	# Compute Y_dot=(X_dot,µ_dot)
	G[0:-1]=0.;
	G[  -1]=1.;
	Y_dot,exit = spla.lgmres(DGy,G, tol = tol_gmres*b_norm, maxiter=250, inner_m = Krylov_Space_Size);
	
	if (exit != 0):
		return 0.*Y_dot,Y,sign,ds, KE,NuT,NuS, True;
	else:	
		Y_dot = Y_dot/np.sqrt( δ*np.dot(Y_dot[0:-1],Y_dot[0:-1]) + (1. - δ)*(Y_dot[-1]**2) );

		return    Y_dot,Y,sign,ds, KE,NuT,NuS, True;
'''

class result():
	"""
	class for result of continuation

	Inputs:
	None

	Returns:
	None
	"""

	def __init__(self):

		self.NuT = []; 
		self.NuS = []; 
		self.KE  = [];
		self.Norm= [];

		self.Ra     = [];
		self.Ra_dot = []; 
		self.Y_FOLD = [];

		self.X_DATA  = [];
		self.Ra_DATA = [];

		self.Iterations = 0;

	def __str__(self):

		s= ( 'Continuation succeed \n'
			+'Total iterations  = '+str(self.Iterations)        	  +'\n'
			+'Ra                = '+str(self.Ra[    self.Iterations]) +'\n'
			+'Ra_dot            = '+str(self.Ra_dot[self.Iterations]) +'\n'
			+'NuT               = '+str(self.NuT[   self.Iterations]) +'\n'
			+'NuS               = '+str(self.NuS[   self.Iterations]) +'\n'
			+'KE                = '+str(self.KE[    self.Iterations]) +'\n'
		);
		return s

def _NewtonC(		 Y  ,sign,ds, **kwargs_f):
	
	"""

	"""

	Iteration = 0;
	while (Iteration < 10):
	
		# Shallow copy as mutable
		X  = Y[0:-1].copy();
		Ra = Y[  -1].copy() + sign*ds;

		kwargs_f["Ra"] = Ra
		X_new,Norm,KE,NuT,NuS,exitcode = _Newton(X,**kwargs_f);

		if exitcode == True:
			ds*=1.5;
			Y_new = np.hstack( (X_new,Ra) )
			return Y_new,sign,ds,	Norm,KE,NuT,NuS,	exitcode;
		else:
			ds*=0.5;
			Iteration+=1;		

	return Y,sign,ds,	Norm,KE,NuT,NuS,	exitcode;

def _Continuation(filename,N_steps,	Y,**kwargs):

	"""

	"""

	Y     = Y.copy()
	Y_dot = np.zeros( Y.shape[0] );

	# Default parameters	
	ds    =.01;	
	sign  =-1.;
	ds_min=0.005;
	ds_max=10.0;

	# file saving
	Result = result()
	# try:
	# 	## Appending onto the old file
	# 	with h5py.File(filename, 'r') as f:
	# 		ff=f["Bifurcation"]
	# 		for key in ff.keys():
				
	# 			if isinstance(ff[key][()], np.ndarray):
	# 				setattr(Result, key, ff[key][()].tolist());
	# 			else:
	# 				setattr(Result, key, ff[key][()] 		 );	
	# 	f.close()

	# 	N_steps+=Result.Iterations;
	# 	Y[0:-1] = Result.X_DATA[ -1];
	# 	Y[  -1] = Result.Ra_DATA[-1];

	# except:
	# 	pass;	
	
	# Main-Loop	
	while (Result.Iterations < N_steps):

		Y_new,sign,ds, Norm,KE,NuT,NuS, exitcode  = _NewtonC(Y,sign,ds, **kwargs);

		# 1) Netwon Solve
		if ds > ds_min:

			Y_new,sign,ds, Norm,KE,NuT,NuS, exitcode  = _NewtonC(Y,sign,ds, **kwargs);	

			# if exitcode == False:
			# 	Y_dot_new, Y_new,sign,ds,	KE,NuT,NuS,	exitcode = _ContinC(Y_dot,Y,sign,ds_min,*args_f);
			# 	if exitcode == False:
			# 		raise ValueError("Neither Netwon nor Psuedo-Arc length converged terminating .... \n")

			if (ds > ds_max):
				ds = ds_max;	

		# # 2) Psuedo Solve		
		# elif ds <= ds_min:

		# 	Y_dot_new, Y_new,sign,ds,	KE,NuT,NuS,	exitcode = _ContinC(Y_dot,Y,sign,ds,*args_f);

		# 	# Saddle detection
		# 	if (Y_dot_new[-1]*Y_dot[-1]) < 0:
		# 		Result.Y_FOLD.append(Y_new);

		# 	# Determine sign for Netwon Step
		# 	if (Y_new[-1] > Y[-1]):
		# 		sign = 1.0;
		# 	else:
		# 		sign = -1.;	

		
		# 3) Update solution
		Y 	  = Y_new.copy(); 	
		#try:
		#	Y_dot = Y_dot_new.copy();
		#except:
		Y_dot =0.*Y;

		# 4) Append & Save data for bifurcation diag
		Result.Ra.append( Y[-1]);
		Result.Ra_dot.append(Y_dot[-1]);
		
		Result.Norm.append(Norm);
		Result.KE.append(  KE  );
		Result.NuT.append( NuT );
		Result.NuS.append( NuS );         
		
		print(Result,flush=True);

		Result.Iterations+=1

		if Result.Iterations%5==0:

			Result.X_DATA.append( Y[0:-1]);
			Result.Ra_DATA.append(Y[  -1]);

			f = h5py.File(filename, 'w');

			Checkpoints = f.create_group("Checkpoints");
			Checkpoints['X_DATA']  = Result.X_DATA;
			Checkpoints['Ra_DATA'] = Result.Ra_DATA;

			Parameters = f.create_group("Parameters");
			for key,val in kwargs.items():
				Parameters[key] = val;

			Bifurcation = f.create_group("Bifurcation");	
			for item in vars(Result).items():
				Bifurcation.create_dataset(item[0], data = item[1])

			f.close()

	return None;	

def Continuation(open_filename,save_filename='ContinuationTest_0.h5',frame=-1):

	"""
	Given an initial condition and full parameter specification perform branch continuation

	Inputs:

	open_filename (.h5) 
	save_filename (.h5)
	frame integer

	Returns:
	None

	"""

	if open_filename.endswith('.h5'):

		f = h5py.File(open_filename, 'r+')

		# Problem Params
		X      = f['Checkpoints/X_DATA'][frame];
		
		try:
			Ra = f['Checkpoints/Ra_DATA'][frame];
		except:
			Ra = f['Parameters']["Ra"][()];

		Ra_s   = f['Parameters']["Ra_s"][()];
		Tau    = f['Parameters']["Tau"][()];
		Pr     = f['Parameters']["Pr"][()];
		d 	   = f['Parameters']["d"][()]

		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]

		f.close();

		print("\n Loading Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(Ra,d,N_fm,N_r))    

		# ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		from Matrix_Operators import INTERP_RADIAL,INTERP_THETAS
		#fac_R =2; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		fac_T =2; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	N_steps= 200;
	Y      = np.hstack( (X,Ra) );
	kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}

	save_filename = uniquify(save_filename)
	
	_Continuation(save_filename,N_steps,Y,**kwargs)

	_plot_bif(save_filename)

	return None;


def _plot_bif(filename):

	obj = result();
	with h5py.File(filename, 'r') as f:
		ff=f["Bifurcation"]
		for key in ff.keys():
			setattr(obj, key, ff[key][()])

	fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,figsize=(12, 6))

	ax0.semilogy(obj.Ra,obj.KE,'k-')
	ax0.set_title(r'Kinetic Energy',fontsize=25)

	ax1.semilogy(obj.Ra,obj.NuT,'k-')
	ax1.set_title(r'$Nu_T-1$',fontsize=25)

	ax2.semilogy(obj.Ra,obj.NuS,'k-')
	ax2.set_title(r'$Nu_S-1$',fontsize=25)
	
	for f in (ax0, ax1, ax2):
		f.set_xlabel(r'$Ra$',fontsize=25)
		f.set_xlim([obj.Ra[0],obj.Ra[-1]])
	
	plt.tight_layout()
	plt.savefig("Bifurcation_Series.pdf",format='pdf', dpi=1200)
	plt.show()        

	return None;

def Plot_full_bif(folder):

	import glob

	fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,figsize=(12, 6))
	ax0.set_title(r'Kinetic Energy',fontsize=25)
	ax1.set_title(r'$Nu_T-1$',fontsize=25)
	ax2.set_title(r'$Nu_S-1$',fontsize=25)
	for f in (ax0, ax1, ax2):
		f.set_xlabel(r'$Ra$',fontsize=25)

	def add_to_fig(obj):
		ax0.semilogy(obj.Ra,obj.KE ,'k-')
		ax1.semilogy(obj.Ra,obj.NuT,'k-')
		ax2.semilogy(obj.Ra,obj.NuS,'k-')
		return None;

	for filename in glob.glob(folder + '/*'):
		
		print(filename)
		
		obj = result();
		with h5py.File(filename, 'r') as f:
			ff=f["Bifurcation"]
			for key in ff.keys():
				setattr(obj, key, ff[key][()])

		add_to_fig(obj)

	plt.tight_layout()
	plt.savefig("Bifurcation_Series.pdf",format='pdf', dpi=1200)
	plt.show()   

	return None;	

# Execute main
if __name__ == "__main__":
	
	# %%
	print("Initialising the code for running...")
	#Time_Step()#file,file,frame);


	#%%
	Newton(open_filename='EigVec.npy',frame=-1);

	# %% 
	print_h5py('NewtonSolve_0.h5')

	from Plot_Tools import Cartesian_Plot, Energy,Uradial_plot
	Cartesian_Plot(filename='NewtonSolve_0.h5',frame=-1,Include_Base_State=False)
	Energy(filename='NewtonSolve_0.h5',frame=-1)
	
	#%%
	Continuation(open_filename='ContinuationTest_1.h5',frame=-1)

	# %%
	Plot_full_bif(folder='/home/pmannix/SpectralDoubleDiffusiveConvection/Branch_l10')

	# %%
	from Plot_Tools import Cartesian_Plot, Energy,Uradial_plot
	Cartesian_Plot(filename='ContinuationTest_3.h5',frame=-1,Include_Base_State=False)
	Energy(filename='ContinuationTest_3.h5',frame=-1)

##	
# %%
