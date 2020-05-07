#! /usr/bin/env python

import numpy as np
print np.__version__
import sys, time
from scipy.sparse.linalg import inv
from EigenVals import *

# For time-stepping
import N1_FT_PARA
N1_FT_PARA.non_lin1.initialize_coeffs()

# For steady-state
import N1
N1.non_lin1.initialize_coeffs()
import N2
N2.non_lin2.initialize_coeffs()

#sys.exit()
#import N1_FT
#from scipy.sparse import csr_matrix
#from Projection	import *



# Time-Stepping Routines
#@profile
def Matrices_Tstep(Tau,Pr,Ra,sigma,D,R,N_modes): # Correct

	#M_INV = np.linalg.inv(M);
	#L0 = L_0(D,R,sigma,Ra,Pr,Tau,N_modes);
	#L = np.matmul(M_INV,L0);	

	M = M_0_SPAR(D,R,N_modes).tocsc()
	L = L_0_SPAR(D,R,sigma,Tau,Pr,Ra,N_modes).tocsc();

	return M,L;

#R@profile
def IMPLICIT(RUNS,Tau,Pr,Ra,sigma,Nr,N_modes,dt,XX): # Correct
	

	#~~~~~~~ Takes a vector DIM (3*Nr*Modes,1) ~~~~~~~~~~~~~~~

	D,R = cheb_radial(Nr,sigma)
	
	# Crank Nicholson Method, Correct
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~
	M,L = Matrices_Tstep(Tau,Pr,Ra,sigma,D,R,N_modes); # Obtain the linear and forcing matrices

	L_left = M - 0.5*dt*L; L_right = M + 0.5*dt*L;
	N_CR = inv(L_left); L_CR = N_CR.dot(L_right);
	

	# Declare All matrices dependent on R
	nr = len(R)-2;
	D3 = np.zeros((nr,nr,N_modes)); 
	D2 = np.zeros((nr,nr,N_modes));
	for ii in xrange(N_modes):
		D3[:,:,ii] = D3_SP(D,R,ii)
		D2[:,:,ii] = Stokes_D2_SP(D,R,ii)

	# Convert to fortran
	D3 = np.asfortranarray(D3);
	D2 = np.asfortranarray(D2);
	D = np.asfortranarray(D);
	R = np.asfortranarray(R);
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~	

	# Declare Vector X	
	NN = 3*nr*N_modes;
	iters = RUNS;
	CAP_STEP = 10; #200;	
	tsteps = iters/CAP_STEP;
	X_VID = np.zeros((NN,tsteps));

	X = np.zeros((NN,1)); X_new = np.zeros((NN,1));
	N_Prev = np.zeros((NN,1)); N_Curr = np.zeros((NN,1));
	N_Curr1 = np.zeros((NN,1))
	
	#DF = np.zeros((NN,NN));

	# Recast X if required
	if len(X) > len(XX):
		X[0:len(XX),0] = XX[:,0]; #.reshape(len(XX));
	elif len(X) < len(XX):
		X[:,0] = XX[0:len(X),0]; #.reshape(len(X));	
	else:
		X = XX;	
	
	err = 1.0; it = 0; ii = 0; X_Norm = [];
	start_time = time.time()
	while ii < iters: # err > 1e-08: #(1e-4):	
		
		# Crank Nicholson, Correct Implemented
		### Ensure Fortran-contiguous arrays are passed
		ARGS_NF = [X,D3,D2,D[1:-1,1:-1],R[1:-1]];#,aPSI,bPSI,aT,bT];
		N_Curr[:,0] = N1_FT_PARA.non_lin1.n1_full(*ARGS_NF)
		#print "Evaluated fine"

		#DF = N1.non_lin1.n1_full(*ARGS_NF)
		#print "Got here"
		#N_Curr1[:,0] = np.matmul(DF,X)[:,0]
		#print "Check diff ",np.allclose(N_Curr,N_Curr1,rtol=1e-12)

		N_0 = 3.0*N_Curr - N_Prev;
		X_new = (dt/2.0)*N_CR.dot(N_0) + L_CR.dot(X);
		N_Prev = N_Curr;

		if ii%CAP_STEP == 0:

			print 'Iteration: ', ii,"\n"

			err = np.linalg.norm((X_new[:,0] - X[:,0]),2)/np.linalg.norm(X_new[:,0],2)				
			print 'Error ',err
			if np.isnan(err): # on nan
				print "Max error exceeded"
				break

			EVE = 0.0; ODD = 0.0;
			for m in xrange(N_modes):
				if m == 0:
					EVE += np.linalg.norm(X[(len(R)-2):3*(len(R)-2),0],2);
				elif m%2 == 0:
					ind = 3*(len(R)-2) + (m-1)*3*(len(R)-2);
					EVE += np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);
				elif m%2 == 1:
					ind = 3*(len(R)-2) + (m-1)*3*(len(R)-2);
					ODD += np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);	
					
			print "EVEN ||X|| ",EVE,"\n"
			print "ODD  ||X|| ",ODD,"\n"

		X_Norm.append(np.linalg.norm(X));
		
		if (it<(tsteps-1)) and (ii%CAP_STEP == 0): 
			#------- ARRAY CAPTURE FOR VIDEO -------
			print "Captured frame n= ",it,"\n"
			X_VID[:,it] = np.squeeze(X_new, axis=1)[:]; 
			it+=1
		
		ii+=1;	
		X = X_new;	

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))		
	'''
	# --------------------------- SAVING AND COMPILING RESULTS -------------------
	if True:
		#1) Change directory into Results file
		#os.chdir('/home/mannixp/Dropbox/Imperial/MResProject/Python_Codes/Transmission/')	
		#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo_SPEC/')
		#os.chdir('~/')
		#Stringy = ['/home/ma/p/pm4615/Pseudo_SPEC/']
		#os.chdir("".join(Stringy))

		# 2) Make new results folder
		st_Pr = "%4.2f"%Pr; st_eps = "%2.2f"%epsilon;
		st_Re1 = "%4.1f"%Re_1; st_Re2 = "%4.1f"%Re_2;
		Stringy = ['Time_Dependent_1C_Sol_Pr',st_Pr,'_epsilon',st_eps,'_Re1',st_Re1,'_Re2',st_Re2];
		os.mkdir("".join(Stringy))

		# 3) Change into Results folder
		os.chdir("".join(Stringy))	

		#Create an associated Parameters file
		Parameters = np.zeros(18)
		file = open("Parameters.txt",'w')
		
		file.write(" #~~~~~~~~~~~~~~~~ Control Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

		Parameters[0] = Ra_c
		file.write('%5.4f'% Ra_c); file.write(" # Ra_c Critical Rayleigh #0 \n");
		
		Parameters[1] = sigma
		file.write('%5.3f'% sigma); file.write(" # d Sigma Gap Width #1 \n")

		Parameters[2] = Re_1
		file.write('%5.4f'% Re_1); file.write(" # Re_1 Inner Reynolds Number #2 \n" )
		Parameters[3] = Re_2
		file.write('%5.4f'% Re_2); file.write(" # Re_2 Outer Reynolds Number #3 \n" )

		Parameters[4] = epsilon
		file.write('%5.4f'% epsilon); file.write(" # epsilon #4 \n")
		
		file.write(" #~~~~~~~~~~~~~~~~ Numerical Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

		Parameters[5] = dt
		file.write('%1.6f' % dt); file.write(" # Time step #5 \n")

		Parameters[6] = len(R)
		file.write('%d' % len(R)); file.write(" # Nr Radial steps #6 \n")
		
		Parameters[7] = N_modes
		file.write('%d' % N_modes);	file.write(" # N_Modes Polar Mode #7 \n")

		Parameters[8] = Pr
		file.write('%5.4f'% Pr); file.write(" # Pr Prandtl nunber #8\n");	

		STR = "".join(['XD_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy'])
		STRVID = "".join(['XD_VID_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy'])

		np.save(STRVID,X_VID)
		np.save(STR,X)
		np.save("PSI_L2.npy",X_Norm)
		np.save("Parameters.npy",Parameters)

		#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo_SPEC/')
		'''

	return X,X_VID;	


# Steady-State Solving Routines	
#@profile
def Matrices(Tau,Pr,Ra,sigma,D,R,N_modes):
 	
 	# Speed up this process!!
	#start_time = time.time()
	L = L_0_SPAR(D,R,sigma,Tau,Pr,Ra,N_modes);
	'''end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))	

	start_time = time.time()
	L0 = L_0(D,R,sigma,Ra,Pr,Tau,N_modes);
	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))	
	
	print np.allclose(L.todense(),L0,rtol=1e-15)

 	plt.spy(L, precision=1e-12,markersize=5)
 	plt.show()
	plt.spy(L0, precision=1e-12,markersize=5)
	plt.show()
	'''
	#L0 = L_0(D,R,sigma,Ra,Pr,Tau,N_modes);
	return L.tocsr()

def Netwon_It(Tau,Pr,Ra,sigma,Nr,N_modes,XX): #'GF = 0.0): # Correct

	D,R = cheb_radial(Nr,sigma)
	L = Matrices(Tau,Pr,Ra,sigma,D,R,N_modes)
	# Use precoditioner also
	
	# Declare All matrices dependent on R
	nr = len(R)-2;
	D3 = np.zeros((nr,nr,N_modes)); D2 = np.zeros((nr,nr,N_modes));
	for ii in xrange(N_modes):
		D3[:,:,ii] = D3_SP(D,R,ii)
		D2[:,:,ii] = Stokes_D2_SP(D,R,ii)
		
	# Declare Vector X	
	NN = 3*nr*N_modes;
	X = np.zeros((NN,1)); 
	X_new = np.zeros(X.shape);
	
	DF1 = np.zeros((NN,NN));
	DF = np.zeros((NN,NN));

	# Recast X if required
	if len(X) > len(XX):
		X[0:len(XX),0] = XX[:,0]#.reshape(len(XX));
	elif len(X) < len(XX):
		X[:,0] = XX[0:len(X),0]#.reshape(len(X));	
	else:
		X = XX;	
	'''
	def mv(X):	
		N1 = N_Full(X,L.shape,N_modes,nr,D3,D2,D,R);
		N2 = N2_Full(X,L.shape,N_modes,nr,D3,D2,D,R);

		return np.matmul(L+N1+N2,X)
	DX = np.zeros(X.shape);
	b = np.matmul(L+N1,X) + F; 
	A = LinearOperator(L.shape, matvec=mv,dtype='float64'); # Operator shape & where it's defined
	SOL = bicgstab(A, b, x0=X, tol=1e-12, maxiter=10**4,M=L_inv);
	DX[:,0] = SOL[0]; # Update X
	X_new = X - DX;
	print "CONVERG: INFO",SOL[1]
	'''

	err = 1.0; it = 0;
	start_time = time.time()
	while err > (1e-9):	

		# Obtain these directly without additional calls
		#N1 = N_Full(X,(NN,NN),N_modes,nr,D3,D2,D,R);
		#N2 = N2_Full(X,(NN,NN),N_modes,nr,D3,D2,D,R);
		
		ARGS_NF = [X,D3,D2,D[1:-1,1:-1],R[1:-1]];
		DF1 = N1.non_lin1.n1_full(*ARGS_NF); # Add N1
		DF = DF1 + N2.non_lin2.n2_full(*ARGS_NF) # Add N2


		F_X = np.matmul(DF1,X) + L.dot(X); 
		X_new = X - np.linalg.solve(L.todense()+DF,F_X);
		
		err = np.linalg.norm((X_new - X),2)/np.linalg.norm(X_new,2)							
		print 'Error ',err
		print 'Iteration: ', it,"\n"

		X = X_new;
		it+=1;

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))		

	#np.save(STR,X)
	return X;	