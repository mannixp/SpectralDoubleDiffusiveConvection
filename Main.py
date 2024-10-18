from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from Matrix_Operators import cheb_radial
import os 
import time
import warnings
import h5py
warnings.simplefilter('ignore', np.RankWarning)


def uniquify(path):
	filename, extension = os.path.splitext(path)

	counter = int(filename.split('_')[1])
	while os.path.exists(path):
		path = filename.split('_')[0] + "_" +str(counter) + extension
		counter += 1

	return path


def Base_State_Coeffs(d):
	"""
	Compute the base state coefficients i.e.

	T_0  = - A_T/r + B_T;

	T'_0 =   A_T/r^2;
	
	"""

	R_1 = 1./d
	R_2 = (1. + d)/d
	A_T = (R_1*R_2)/(R_1 - R_2)
	B_T = R_1/(R_1 - R_2)

	return A_T, B_T


@njit(fastmath=True)
def Nusselt(T_hat, d, R, D, N_fm, nr):

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


def Kinetic_Energy(X_hat, R, D, N_fm, nr):
	"""
	Compute the volume integrated kinetic energy

	KE = (1/2)*(1/V) int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ

	where

	V = int_r1^r2 int_0^π KE(r,θ) r^2 sin(θ) dr dθ = (2/3)*(r2^3 - r1^3)
	
	and

	KE = |u_r|^2 + |u_θ|^2
	
	where

	u_r = 1/(r^2 sin θ) ∂(ψsinθ)/∂θ = (1/r^2) J_θ(ψ) ~ cosine

	u_θ =  -(1/r) ∂ψ/∂r 			=  -(1/r) D_r(ψ) ~ sine

	when substituted gives

	KE(r,θ) r^2 sin(θ) = [ (J_θ(ψ)/r)^2 + (D_r(ψ))^2 ]*sin(θ)

	"""

	from Matrix_Operators import J_theta_RT
	from Transforms import IDCT,IDST,grid

	N  = N_fm*nr;
	Dr = D[1:-1,1:-1];
	IR = np.diag(1./R[1:-1]);

	ψ_hat=X_hat[0:N]
	JPSI =J_theta_RT(ψ_hat, nr,N_fm)

	Jψ_hat = np.zeros((nr, N_fm)); 	
	Dψ_hat = np.zeros((nr, N_fm)); 
	for k in range(N_fm):
		ψ_k          = ψ_hat[k*nr:(1+k)*nr];
		Dψ_hat[:,k]  = Dr@ψ_k                 # ~ sine
		Jψ_hat[:,k]  = IR@JPSI[k*nr:(1+k)*nr] # ~ cosine
	
	# Convert Sine to sinusoids
	Dψ_hat[:,1:] = Dψ_hat[:,0:-1]; Dψ_hat[:,0] = 0.0;

	θ   = grid(3*N_fm)
	Jψ  = IDCT(Jψ_hat,n = 3*N_fm) 
	Dψ  = IDST(Dψ_hat,n = 3*N_fm)

	KE_rθ = Jψ**2  +  Dψ**2;
	
	# sin_θ = np.outer(np.ones(nr),np.sin(θ))
	# KE_r  = np.trapz(KE_rθ*sin_θ,x=θ      ,axis=1)
	# KE    = np.trapz(KE_r       ,x=R[1:-1],axis=0);
	# f  = 0*R; f[1:-1] = KE_r[:];
	# KE = np.linalg.solve(D[0:-1,0:-1],f[0:-1])[0]; print('KE spectral = ',KE)

	KE_θ = np.trapz(KE_rθ         ,x=R[1:-1],axis= 0)
	KE   = np.trapz(KE_θ*np.sin(θ),x=θ      ,axis=-1)

	V    = (2./3.)*(R[-1]**3 - R[0]**3);

	return (.5/V)*KE;


def Eq_SYM(X1, R):

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


def Build_Matrix_Operators(N_fm, N_r, d, dt):

	"""
	Builds all the matrix operators that have a radial dependence
	and returns these as a tuple of positional arguments

	Inputs:

	N_fm - (int) number of latitudinal modes
	N_r  - (int) number of radial collocation points
	d 	 - (float) gap width

	"""
	
	from Matrix_Operators import R2, kGR_RT
	from Matrix_Operators import NAB2_TSTEP_MATS, A4_TSTEP_MATS

	D, R = cheb_radial(N_r, d) 
	Rsq = R2(R, N_fm)
	gr_K = kGR_RT(R, N_fm, d) 

	# For T0_J_Theta, as a vector!!!
	A_T = Base_State_Coeffs(d)[0]
	DT0 = A_T/(R[1:-1]**2)

	nr = N_r - 1
	args_FX = (D, R, N_fm, nr)
	L_inv_NAB2 = NAB2_TSTEP_MATS(dt, N_fm, nr, D, R)
	L_inv_A4 = A4_TSTEP_MATS(dt, N_fm, nr, D, R)

	return D, R, Rsq, DT0, gr_K, L_inv_A4, L_inv_NAB2, args_FX


def _Time_Step(X, Ra, Ra_s, Tau, Pr, d,	N_fm, N_r, save_filename='TimeStep_0.h5', start_time=0, Total_time=1, dt=1e-04, symmetric=True, linear=False, Verbose=False):

	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta, A2_SINE	
	from Matrix_Operators import A4_BSub_TSTEP_V2, NAB2_BSub_TSTEP_V2
	D, R, Rsq, DT0, gr_k, L_inv_A4, L_inv_NAB2, args_FX = Build_Matrix_Operators(N_fm, N_r, d, dt)

	Time = []
	X_DATA = []
	KE = []  
	NuT = [] 
	NuS = []
	Norm = []

	if symmetric is True:
		X_SYM = Eq_SYM(X, R)
	else:
		X_SYM = 1

	nr = len(R[1:-1])
	error     = 1.0;
	iteration = 0;
	N_iters   = int(Total_time/dt);
	N_save    = int(N_iters/10);
	N_print   = 10;
	N         = int(X.shape[0]/3)

	def Step_Python(Xn, linear):

		ψ = Xn[0:N]
		T = Xn[N:2*N]
		S = Xn[2*N:3*N]

		if linear is False:
			NX = -dt*FX(Xn, *args_FX, symmetric); # 36% FIX
		else:
			NX = np.zeros(3*N)
		
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric)
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric)

		# 1) Vorticity - Ω
		# Here we invert the LHS = ( \hat{A}^2 − ∆t Pr \hat{A}^4)
		NX[0:N]     += Ω   + dt*Pr*gr_k.dot(Ra*T - Ra_s*S)
		ψ_new        = A4_BSub_TSTEP_V2(NX[0:N],  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric)

		# 2) Temperature - T
		# Here we invert the LHS = r^2( I − ∆t \hat{\nabla}^2)
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0
		T_new 		 = NAB2_BSub_TSTEP_V2(NX[N:2*N], L_inv_NAB2, N_fm, nr, dt, symmetric)

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0
		S_new		 = NAB2_BSub_TSTEP_V2(NX[2*N:3*N], L_inv_NAB2, N_fm, nr, Tau*dt, symmetric)
		
		return np.hstack((ψ_new,T_new,S_new)),KE;

	timer = 0.	
	while iteration < N_iters:

		ST_time = time.time()
		X_new,kinetic = Step_Python(X,linear);
		END_time = time.time();
		
		Norm.append( np.linalg.norm(X_new,2) );
		KE.append(  Kinetic_Energy(X_new,    R,D,N_fm,nr) );
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


def Time_Step(open_filename=None, frame=-1, delta_Ra=0):

	"""
	Given an initial condition and full parameter specification time-step the system
	using time-stepping scheme CNAB1

	Inputs:

	Returns:

	"""

	# ~~~~~# L = 11 Gap #~~~~~~~~~#
	# Ra = 9775.905436191546 # steady
	Ra  = 2879.0503253066827 # hopf
	d = 0.31325;
	
	# ~~~~~~~~~~~~~~~ Gap widths l=10~~~~~~~~~~~~~~~
	# l10
	#Ra	   = 9851.537357677651 - 1e-03
	# Ra   = 2965.1798389922933
	# d	   = 0.3521

	Ra_s   = 500.
	Tau    = 1./15.;
	Pr     = 1.

	N_fm = 32;
	N_r  = 15;

	N_fm_n = 32;
	N_r_n  = 15;

	# ~~~~~~~~~ Random Initial Conditions ~~~~~~~~~~~~~~~~
	N = (N_r - 1)*N_fm;
	X = np.random.rand(3*N);
	X = 1e-03*(X/np.linalg.norm(X,2))
	
	# ~~~~~~~~~ Initial Conditions ~~~~~~~~~~~~~~~~~~
	if open_filename.endswith('.h5'):

		f = h5py.File(open_filename, 'r+')

		# Problem Params
		X = f['Checkpoints/X_DATA'][frame];
		#Ra = f['Checkpoints/Ra_DATA'][frame];
		Ra = f['Parameters']["Ra"][()];
		# Fix these they should be the same
		# try:
		# 	Ra = f['Bifurcation/Ra_DATA'][frame];
		# except:
		# 	Ra = f['Parameters']["Ra"][()];

		Ra_s   = f['Parameters']["Ra_s"][()];
		Tau    = f['Parameters']["Tau"][()];
		Pr     = f['Parameters']["Pr"][()];
		d 	   = f['Parameters']["d"][()]

		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]

		st_time= 0#f['Scalar_Data/Time'][()][frame]
		f.close();    

		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))    

		# ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		N_r_n  = 20;  X = INTERP_RADIAL(N_r_n, N_r, X, d)
		N_fm_n = 256; X = INTERP_THETAS(N_fm_n, N_fm, X)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	try:
		filename = uniquify(open_filename)
	except:
		filename = uniquify('TimeStep_0.h5')

	Ra = Ra + delta_Ra;
	print('Ra simulation = %e \n'%Ra)
	kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm_n,"N_r":N_r_n}
	X_new  = _Time_Step(X,**kwargs,save_filename=filename,start_time=0,Total_time=100,dt=2.5e-03,symmetric=False,linear=False,Verbose=True);

	return filename;


def _Newton(X, Ra, Ra_s, Tau, Pr, d, N_fm, N_r, symmetric=False, dt=10**4, tol_newton=1e-08, tol_gmres=1e-04, Krylov_Space_Size=150):
	
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
	from Matrix_Operators import DT0_theta, A2_SINE
	from Matrix_Operators import A4_BSub_TSTEP_V2, NAB2_BSub_TSTEP_V2
	
	D, R, Rsq, DT0, gr_k, L_inv_A4, L_inv_NAB2, args_FX = Build_Matrix_Operators(N_fm, N_r, d, dt)

	nr = N_r -1
	N  = N_fm*nr 
	dv = np.random.randn(X.shape[0])

	error     = 1.0; 
	iteration = 0; 
	exit      = 1.0; 

	if symmetric is True:
		X_SYM = Eq_SYM(X, R)
	else:
		X_SYM = 1

	def PFX(Xn):
		
		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# Linear
		NX    = -1.*dt*FX(Xn, *args_FX, symmetric); # 36% FIX
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		NX[0:N]     += Ω + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        = A4_BSub_TSTEP_V2(NX[0:N],  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric) - ψ;

		# 2) Temperature - T
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new 		 = NAB2_BSub_TSTEP_V2(NX[N:2*N], L_inv_NAB2, N_fm, nr, dt, symmetric) - T;

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new		 = NAB2_BSub_TSTEP_V2(NX[2*N:3*N], L_inv_NAB2, N_fm, nr, Tau*dt, symmetric) - S;

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
		ψ_new        = A4_BSub_TSTEP_V2(NX[0:N],  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric) - δψ;

		# 2) Temperature - ∆T
		NX[N:2*N]   += Rsq.dot(δT) - dt*δψ_T0;
		T_new 		 = NAB2_BSub_TSTEP_V2(NX[N:2*N], L_inv_NAB2, N_fm, nr, dt, symmetric) - δT;

		# 3) Solute - ∆S
		NX[2*N:3*N] += Rsq.dot(δS) - dt*δψ_T0;
		S_new		 = NAB2_BSub_TSTEP_V2(NX[2*N:3*N], L_inv_NAB2, N_fm, nr, Tau*dt, symmetric) - δS;

		return np.hstack((ψ_new,T_new,S_new));	

	while (error > tol_newton) and (iteration < 5):
		
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

	if (iteration == 5) or (exit != 0):

		print('Newton iteration could not converge');
		return None,None,None,None,None, False;
	
	else:	
		
		# Compute diagnoistics
		Norm = np.linalg.norm(X,2);
		KE   = Kinetic_Energy(X,    R,D,N_fm,nr);
		NuT  = Nusselt(X[N:2*N]  ,d,R,D,N_fm,nr);
		NuS  = Nusselt(X[2*N:3*N],d,R,D,N_fm,nr);

		return X,	Norm,KE,NuT,NuS, True;


def Newton(fac, open_filename='NewtonSolve_0.h5', save_filename='NewtonSolve_0.h5', frame=-1):

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
		Ra     = f['Checkpoints/Ra_DATA'][frame];

		#Ra     = f['Parameters']["Ra"][()];
		Ra_s   = f['Parameters']["Ra_s"][()];
		Tau    = f['Parameters']["Tau"][()];
		Pr     = f['Parameters']["Pr"][()];
		d 	   = f['Parameters']["d"][()]

		N_fm   = f['Parameters']["N_fm"][()]
		N_r    = f['Parameters']["N_r"][()]

		try:
			st_time= f['Scalar_Data/Time'][()][frame]
		except:
			st_time = 0
		f.close();

		symmetric = False
		print("\n Loading time-step %e with parameters Ra = %e, d=%e and resolution N_fm = %d, N_r = %d \n"%(st_time,Ra,d,N_fm,N_r))    

		# # ~~~~~~~~~ Interpolate ~~~~~~~~~~~~~~~~~~~
		# from Matrix_Operators import INTERP_RADIAL, INTERP_THETAS
		# fac_R =1; X = INTERP_RADIAL(int(fac_R*N_r),N_r,X,d);  N_r  = int(fac_R*N_r);
		# fac_T =1; X = INTERP_THETAS(int(fac_T*N_fm),N_fm,X);  N_fm = int(fac_T*N_fm)
		# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	# ~~~~~~~~~ Old Initial Conditions ~~~~~~~~~~~~~~~~~~
	if open_filename.endswith('.npy'):
		X  = np.load(open_filename);

		# ~~~~~# L = 11 Gap #~~~~~~~~~#
		#d = 0.31325; l=11.0; 
		#Ra = 9775.905436191546 # steady
		#Ra  = 2879.0503253066827 # hopf

		# ~~~~~# L = 10 Gap #~~~~~~~~~#
		# d = 0.3521; l=10.0; 
		# Ra = 9851.537357677651; # steady
		# Ra = 2965.1798389922933; # hopf

		d = 0.36737
		l = 10
		Ra = 9889.253292035168


		Ra    -=1e-02
		Ra_s   = 500.
		Tau    = 1./15.;
		Pr     = 1.
		
		N_fm   = 64; #300
		N_r    = 20; #30

		st_time= 0.

		X *= fac*np.linalg.norm(X,2)

		if l%2 ==1:
			symmetric = False
		else:
			symmetric = True

	#~~~~~~~~~~#~~~~~~~~~~#
	# Run Code
	#~~~~~~~~~~#~~~~~~~~~~#
	kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm,"N_r":N_r}
	X_new,Norm,KE,NuT,NuS,_ = _Newton(X,**kwargs,symmetric = symmetric,tol_newton = 5e-8,tol_gmres = 1e-04,Krylov_Space_Size = 250)

	
	#~~~~~~~~~~#~~~~~~~~~~#
	# Save data
	#~~~~~~~~~~#~~~~~~~~~~# 
	if (save_filename == None) and open_filename.endswith('.h5'):
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

	from Plot_Tools import Cartesian_Plot, Energy
	Cartesian_Plot(filename,frame=-1,Include_Base_State=False)
	Energy(filename,frame=-1)

	return None;


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


def _NewtonC(Y, sign, ds, **kwargs_f):
	
	"""

	"""

	# Iteration = 0;
	# while (Iteration < 2):
	
	# Shallow copy as mutable
	X  = Y[0:-1].copy();
	Ra = Y[  -1].copy() + sign*ds;

	kwargs_f["Ra"] = Ra
	X_new,Norm,KE,NuT,NuS,exitcode = _Newton(X,**kwargs_f, tol_newton=1e-08);

	if exitcode == True:
		ds*=2;
		Y_new = np.hstack( (X_new,Ra) )
		return Y_new,sign,ds,	Norm,KE,NuT,NuS,	exitcode;
	else:
		ds*=0.5;
		return Y    ,sign,ds,	Norm,KE,NuT,NuS,	exitcode;


def _ContinC(Y_0_dot, Y_0, sign, ds, Ra, Ra_s, Tau, Pr, d, N_fm, N_r, symmetric=False, dt=10**4, tol_newton=1e-08, tol_gmres=1e-04, Krylov_Space_Size=150):

	"""
	Given a starting point and a control parameter compute a new steady-state using Newton iteration

	Inputs: ??

	defaults
	tol_newton - (float) newton iteration tolerance to terminate iterations
	tol_gmres  - (float) GMRES x =A^-1 b solver algorithim parameter
		#if np.linalg.norm(Y_dot,2) == 0:
	Krylov_Space_Size - (int) number of Krylov vectors to use when forming the subspace

	Returns: ??
	"""

	import scipy.sparse.linalg as spla # Used for Gmres Netwon Sovler

	from Matrix_Operators import NLIN_DFX as DFX
	from Matrix_Operators import NLIN_FX as FX
	from Matrix_Operators import DT0_theta,A2_SINE
	from Matrix_Operators import A4_BSub_TSTEP_V2, NAB2_BSub_TSTEP_V2
	
	D, R, Rsq, DT0, gr_k, L_inv_A4, L_inv_NAB2, args_FX = Build_Matrix_Operators(N_fm, N_r, d, dt)

	nr = N_r - 1
	N  = N_fm*nr 
	dv = np.random.randn(3*N)
	δ  = 1./(3.*N)
	#δ  = 0.01

	if symmetric is True:
		X_SYM  = Eq_SYM(dv,R)
	else:
		X_SYM  = 1.
	
	def PFX(Xn,Ra):
		
		ψ = Xn[0:N];
		T = Xn[N:2*N];
		S = Xn[2*N:3*N];

		# Linear
		NX    = -1.*dt*FX(Xn, *args_FX, symmetric); # 36% FIX
		ψ_T0  = DT0_theta(ψ,   DT0,N_fm,nr, symmetric);
		Ω     = A2_SINE(ψ,     D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - Ω
		NX[0:N]     += Ω + dt*Pr*gr_k.dot(Ra*T - Ra_s*S);
		ψ_new        = A4_BSub_TSTEP_V2(NX[0:N],  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric) - ψ;

		# 2) Temperature - T
		NX[N:2*N]   += Rsq.dot(T) - dt*ψ_T0;
		T_new 		 = NAB2_BSub_TSTEP_V2(NX[N:2*N], L_inv_NAB2, N_fm, nr, dt, symmetric) - T;

		# 3) Solute - S
		NX[2*N:3*N] += Rsq.dot(S) - dt*ψ_T0;
		S_new		 = NAB2_BSub_TSTEP_V2(NX[2*N:3*N], L_inv_NAB2, N_fm, nr, Tau*dt, symmetric) - S;

		return np.hstack((ψ_new,T_new,S_new));

	def PDFX(dv,Xn,Ra):

		δψ = dv[0:N];
		δT = dv[N:2*N];
		δS = dv[2*N:3*N];

		# Linear
		NX     = -1.*dt*DFX(dv,Xn, *args_FX,     symmetric); # 36% FIX
		δψ_T0  = DT0_theta(δψ,   	DT0,N_fm,nr, symmetric);
		δΩ     = A2_SINE(δψ,     	D,R,N_fm,nr, symmetric); 

		# 1) Vorticity - ∆Ω
		NX[0:N]     += δΩ + dt*Pr*gr_k.dot(Ra*δT - Ra_s*δS);
		ψ_new        = A4_BSub_TSTEP_V2(NX[0:N],  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric) - δψ;

		# 2) Temperature - ∆T
		NX[N:2*N]   += Rsq.dot(δT) - dt*δψ_T0;
		T_new 		 = NAB2_BSub_TSTEP_V2(NX[N:2*N], L_inv_NAB2, N_fm, nr, dt, symmetric) - δT;

		# 3) Solute - ∆S
		NX[2*N:3*N] += Rsq.dot(δS) - dt*δψ_T0;
		S_new		 = NAB2_BSub_TSTEP_V2(NX[2*N:3*N], L_inv_NAB2, N_fm, nr, Tau*dt, symmetric) - δS;

		return np.hstack((ψ_new,T_new,S_new));

	def PDFµ(Xn):
		
		T = Xn[N:2*N];

		# DF(X,µ)_µ
		dfµ       = 0.*Xn;
		fµ		  = dt*Pr*gr_k.dot(T)	
		dfµ[0:N]  = A4_BSub_TSTEP_V2(fµ,  L_inv_A4, D, R, N_fm, nr, Pr*dt, symmetric)
	
		return dfµ;

	def Predict(Y_dot,Y,sign,ds):

		# Shallow copy as mutable
		X_0  = X_SYM*Y[0:-1].copy();
		µ_0  = Y[  -1].copy();

		if (np.linalg.norm(Y_dot,2) == 0) or (dt < 10):
			
			# P*DF(X,µ)_µ	
			dfµ  = (-1.)*PDFµ(X_0)

			# P*DF(X,µ)_X
			DF_X = lambda dv: PDFX(dv,X_0,µ_0);
			dfx  = spla.LinearOperator((3*N,3*N),matvec=DF_X,dtype='float64');

			# a) Solve pre-conditioned P*DF_X(X_0)*ξ = (-1)*P*DF_µ(X); 
			b_norm 	= np.linalg.norm(dfµ,2);
			ξ,exit  = spla.lgmres(dfx,dfµ,	maxiter=250, inner_m = Krylov_Space_Size, atol = tol_newton*b_norm);

			if (exit != 0): raise ValueError("\n LGMRES couldn't converge when predicting ξ \n")
			
			# b) Compute x_dot, µ_dot 
			µ_dot = sign/np.sqrt( 1.0 + δ*(np.linalg.norm(ξ,2)-1.0) );
			X_dot = µ_dot*ξ; 
			print('||ξ|| =',np.linalg.norm(ξ,2))
		else:

			# Shallow copy as mutable
			X_dot = Y_0_dot[0:-1].copy(); 
			µ_dot = Y_0_dot[  -1].copy();
		
		# Make prediction	
		X = X_0 + X_dot*ds; 
		µ = µ_0 + µ_dot*ds;
		
		#print('Continuation ds,mu_dot = %e,%e'%(ds,µ_dot),'\n')

		return np.hstack( (X,µ) ), X_0, µ_0, X_dot, µ_dot;
	
	Y, X_0, µ_0, X_dot, µ_dot = Predict(Y_0_dot,Y_0,sign,ds) 
	
	G 			= 0.*Y;
	err_X,err_µ = 1.0,1.0; 
	iteration   = 0; 
	exit        = 1.0; 

	while ( (err_X > tol_newton) or (err_µ > tol_newton) ) or (iteration < 2) :
		
		# )  ~~~~~~~~~~~~~~~~~  ds control  ~~~~~~~~~~~~~~~~~ 
		if (iteration >= 5):
			
			print('Reducing the step-size ds_old=%e -> ds_new=%e \n'%(ds,.5*ds));

			iteration=0;
			ds*=0.5;

			if ds < tol_newton: raise ValueError("\n Step-size ds has become too small terminating \n")
			
			X = X_0 + X_dot*ds; 
			µ = µ_0 + µ_dot*ds;	
			Y = np.hstack( (X,µ) );
	
		X  = X_SYM*Y[0:-1].copy(); 
		µ  = 	   Y[  -1].copy();
		
		# P*DF(X,µ)_X & P*F(X,µ)_µ
		DF_X    = lambda dv: PDFX(dv,X,µ);
		DF_µ    =            PDFµ(X)
		
		# F(X,µ) & p(X,µ,s) 
		G[0:-1] = PFX(X,µ)
		G[  -1] = δ*np.dot(X_dot,X - X_0)  + (1.-δ)*µ_dot*(µ - µ_0) - ds;
		
		# 3) Compute DG_y
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~
		DG  = lambda dY: np.hstack( ( DF_X(          dY[0:-1]) 		   + DF_µ*dY[-1], \
									  δ*np.dot(X_dot,dY[0:-1]) + (1.-δ)*µ_dot*dY[-1] ) );
		
		DGy = spla.LinearOperator((Y.shape[0],Y.shape[0]),matvec=DG,dtype='float64');

		b_norm  = np.sqrt( δ*np.dot(G[0:-1],G[0:-1]) + (1.-δ)*(G[-1]**2) );
		dY,exit = spla.lgmres(DGy,G, maxiter=250, inner_m = Krylov_Space_Size, atol = tol_gmres*b_norm);
		Y 		= Y - dY;
		
		if (exit != 0): raise ValueError("\n LGMRES couldn't converge when predicting dY \n")

		# 4) ~~~~~~~~~~~~~~~~~ Compute error ~~~~~~~~~~~~~~~~~
		err_X = np.linalg.norm(dY[0:-1],2)/np.linalg.norm(X,2);
		err_µ = abs(dY[-1])/abs(µ);
		
		print('Psuedo-Arc Iteration = %d, Error X = %e, Error µ = %e'%(iteration,err_X,err_µ),"\n") 
		iteration+=1;


	if (iteration <= 4):
		print('Increasing the step-size ds_old=%e -> ds_new=%e \n'%(ds,2*ds));
		ds*=2;

	# Compute diagnoistics
	X    = Y[0:-1].copy()
	Norm = np.linalg.norm(X,2);
	KE   = Kinetic_Energy(X,    R,D,N_fm,nr);
	NuT  = Nusselt(X[N:2*N]  ,d,R,D,N_fm,nr);
	NuS  = Nusselt(X[2*N:3*N],d,R,D,N_fm,nr);	

	# Compute Y_dot=(X_dot,µ_dot)
	G[0:-1]=0.
	G[  -1]=1.
	b_norm = np.linalg.norm(G,2);
	Y_dot,exit = spla.lgmres(DGy,G, maxiter=250, inner_m = Krylov_Space_Size, atol = tol_newton*b_norm);
	
	if (exit != 0):
		return 0.*Y_dot,Y,sign,ds, Norm,KE,NuT,NuS, True;
	else:	
		Y_dot = Y_dot/np.sqrt( δ*np.dot(Y_dot[0:-1],Y_dot[0:-1]) + (1. - δ)*(Y_dot[-1]**2) );

		return    Y_dot,Y,sign,ds, Norm,KE,NuT,NuS, True;


def _Continuation(filename, N_steps, sign, Y, **kwargs):

	"""

	"""

	Y     = Y.copy()
	Y_dot = 0.*Y;

	# Default parameters	
	ds    =0.01; # The starting step size	
	ds_min=1.0; #Threshold to switching between Newton & Psuedo
	ds_max=10.0; # Max Newton step

	Result = result()
	
	while (Result.Iterations < N_steps):

		if ds > ds_min:

			OUT      = _NewtonC(Y,sign,ds, **kwargs);	
			exitcode = OUT[-1]

			if exitcode == False:
				print('Switching to arc-length \n')
				Y_dot_new, Y_new,sign,ds,	Norm, KE,NuT,NuS,	exitcode = _ContinC(Y_dot,Y,sign,ds,**kwargs);
			else:
				Y_new,sign,ds, Norm,KE,NuT,NuS = OUT[0],OUT[1],OUT[2], OUT[3],OUT[4],OUT[5],OUT[6]

			if (ds > ds_max):
				ds = ds_max;	

		elif ds <= ds_min:

			Y_dot_new, Y_new,sign,ds,	Norm, KE,NuT,NuS,	exitcode = _ContinC(Y_dot,Y,sign,ds,**kwargs);
			

			# Saddle detection
			if (Y_dot_new[-1]*Y_dot[-1]) < 0:
				Result.Y_FOLD.append(Y_new);

			# Determine sign for Netwon Step
			if (Y_new[-1] > Y[-1]):
				sign = 1.0;
			else:
				sign = -1.;	

		# 3) Update solution
		Y = Y_new.copy(); 	
		try:
			Y_dot = Y_dot_new.copy();
		except:
			Y_dot =0.*Y;

		# 4) Append & Save data
		Result.Ra.append( Y[-1]);
		Result.Ra_dot.append(Y_dot[-1]);
		
		Result.Norm.append(Norm);
		Result.KE.append(  KE  );
		Result.NuT.append( NuT );
		Result.NuS.append( NuS );         
		
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

		print(Result,flush=True);
		Result.Iterations+=1

	return None;	


def Continuation(open_filename, frame=-1):

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
		X  = f['Checkpoints/X_DATA'][frame];
		
		
		# Fix these they should be the same
		try:
			Ra = f['Checkpoints/Ra_DATA'][frame];
			#Ra = f['Bifurcation/Ra_DATA'][frame];
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
		N_r_n  = 20; X = INTERP_RADIAL(N_r_n,N_r,X,d);
		N_fm_n = 128; X = INTERP_THETAS(N_fm_n,N_fm,X);
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	sign   = -1;
	N_steps= 1000;
	Y      = np.hstack( (X,Ra) );
	kwargs = {"Ra":Ra,"Ra_s":Ra_s,"Tau":Tau,"Pr":Pr,"d":d,"N_fm":N_fm_n,"N_r":N_r_n, "symmetric":True}

	save_filename = uniquify(open_filename)
	
	_Continuation(save_filename,N_steps,sign,Y,**kwargs)

	_plot_bif(save_filename)

	return None;


def trim(filename, point):

	# (1) Create a new file with trimmed
	def uniquify_trim(path):
		filename, extension = os.path.splitext(path)
		counter = 0

		while os.path.exists(path):
			path = filename + "_trimmed_" +str(counter) + extension
			counter += 1
		return path

	f_new = h5py.File(uniquify_trim(filename), 'w');

	# (2) Pass the old data into an object
	Result = result();
	f_old  = h5py.File(filename, 'r')

	for key,val in f_old["Bifurcation"].items():
		if   key == "Iterations":
			setattr(Result, key, val[()] + 5*point)
		elif key == "X_DATA" or key == "Ra_DATA":
			setattr(Result, key, val[()][ :point])
		else:
			setattr(Result, key, val[()][ :5*point])

	# (3) Write it to the trimmed file
	Checkpoints = f_new.create_group("Checkpoints");
	Checkpoints['X_DATA']  = Result.X_DATA[ :point];
	Checkpoints['Ra_DATA'] = Result.Ra_DATA[:point];

	Bifurcation = f_new.create_group("Bifurcation");	
	for item in vars(Result).items():
		Bifurcation.create_dataset(item[0], data = item[1])

	Parameters = f_new.create_group("Parameters");
	for key,val in f_old["Parameters"].items():
		Parameters[key] = val[()];

	f_new.close()
	f_old.close()
	
	return None;


def _plot_bif(filename, point = -1):

	obj = result();
	with h5py.File(filename, 'r') as f:
		ff=f["Bifurcation"]
		for key in ff.keys():
			setattr(obj, key, ff[key][()])
		
		# find the closest Ra_data points
		N_fm = f['Parameters']["N_fm"][()];
		N_r  = f['Parameters']["N_r"][()];
		d    = f['Parameters']["d"][()];

	# Locate the saddle nodes
	idx = np.where(obj.Ra_dot[:-1] * obj.Ra_dot[1:] < 0 )[0] +1

	fig = plt.figure()
	plt.semilogy(obj.Ra,obj.KE,'k-')
	plt.semilogy(obj.Ra[idx],obj.KE[idx],'ro')
	
	D,R = cheb_radial(N_r,d); 
	KE = Kinetic_Energy(obj.X_DATA[point], R,D,N_fm,N_r-1)
	plt.semilogy(obj.Ra[point*5],obj.KE[point*5],'y*')
	plt.semilogy(obj.Ra_DATA[point], KE ,'bs')

	plt.title(r'Kinetic Energy',fontsize=25)
	plt.ylabel(r'$\mathcal{E}$',fontsize=25)
	plt.xlabel(r'$Ra$',fontsize=25)
	
	plt.tight_layout()
	plt.savefig("Bifurcation_Series.png",format='png', dpi=200)
	plt.show()        

	return None;


# Execute main
if __name__ == "__main__":

	# %%
	print("Initialising the code for running...")

	# %%
	#Continuation(open_filename='NewtonSolveMinus_0.h5',frame=-1)
	#trim(filename='NewtonSolve_5.h5',point=45)
	#_plot_bif(filename='ContinuationMinus_0.h5',point=-35) #  Good start point
	_plot_bif(filename='ContinuationPlus_0.h5',point=30) #  Good start point

	# %%
	#filename = 'EigVec_l10.npy'
	#Newton(fac=-1e-02,open_filename=filename,frame=-1);

	# %%
	#from Plot_Tools import Cartesian_Plot, Energy,Uradial_plot
	#_plot_bif(filename='ContinuationL11Large_8.h5',point=-1)
	#Cartesian_Plot(filename='ContinuationL11Large_8.h5',frame=-1,Include_Base_State=False)
	#Energy(filename='ContinuationL11Large_8.h5',frame=-1)

# %%
