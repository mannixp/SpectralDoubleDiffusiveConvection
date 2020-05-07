#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys, time, scipy
from scipy.sparse.linalg import inv

from EigenVals import *
from Projection import *
from Routines import *

#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------


## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
##~~~~~~~~~~~~~~~~ Preconditioning based on Tuckerman papaer included ~~~~~~~

#@profile # BOTTEL NECK			
def Matrices_Psuedo(Tau,Pr,Ra,sigma,D,R,N_modes,Y): # Correct

	# Extract X & Ra
	mu = Y[-1,0]; Ra = mu; X = Y[0:-1]; nr = len(R)-2;
	
	D3 = np.zeros((nr,nr,N_modes)); 
	D2 = np.zeros((nr,nr,N_modes));
	for ii in xrange(N_modes):
		D3[:,:,ii] = D3_SP(D,R,ii)
		D2[:,:,ii] = Stokes_D2_SP(D,R,ii)


	# Part a) Linear Operator
	L = L_0_SPAR(D,R,sigma,Tau,Pr,Ra,N_modes);
	M = M_0_SPAR(D,R,N_modes) 
	M_inv = inv(M.tocsc() );
	
	# Part b) Non-Linear Terms
	N1 = N_Full(X,L.shape,N_modes,nr,D3,D2,D,R);
	N2 = N2_Full(X,L.shape,N_modes,nr,D3,D2,D,R);

	# Aim to make L sparse
	#Return I, N1, N2, F all multiplied by M_inv and L_inv

	return M_inv,L.todense(),N1,N2;

def Evo_FF(L,L_inv,FF,N1,N2,Y): # Correct

	# a) Extract XX
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];

	# b) New Solution iterate is
	X = np.matmul(L+N1,XX) + FF;

	# c0 L_inv* for preconditioning
	return np.matmul(L_inv,X); # System state, Type Col Vector X

def Evo_FF_X(L,L_inv,FF,N1,N2,Y): # Correct 
	
	# a) Linearised operator & L_inv* for preconditioning
	F_x = np.matmul(L_inv,L+N1+N2);

	####print "Determinant ",np.linalg.det(F_x),"\n"
	#print "Bialternate Matrix ",det(F_x),"\n"

	return F_x; # Jacobian wtr X @ X_0 Type Matrix_Operator dim(X) * dim(X)

def Evo_FF_mu(L_Ra,L_inv,Y): # Correct
	# \mu may be defined as Re_i or Ra

	# a) Extract XX
	#mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	
	# b) Multiply to set all parts independent of T to zero in XX 		
	
	return np.matmul(L_inv,np.matmul(L_Ra,Y[0:-1])); # Jacobian wrt mu @ X_0 Type Col Vector


## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
#@profile
def xi(L,L_inv,FF,N1,N2,Y):
	args = [L,L_inv,FF,N1,N2,Y]
	
	Fx,F_mu = Evo_FF_X(*args),Evo_FF_mu(L_Ra,L_inv,Y);
	
	return (-1.0)*np.linalg.solve(Fx,F_mu);

def mu_dot(chi):
	
	# derivative of N wrt mu @ mu_0, Type Scalar
	return sign/np.sqrt(1.0 + delta*( pow(np.linalg.norm(chi,2),2) -1.0) );

def X_dot(chi):

	# derivative of N wrt X @ X_0, Type Col Vector
	return mu_dot(chi)*chi; 

#@profile
def D_dots(chi):

	mu_dot = sign/np.sqrt(1.0 + delta*( pow(np.linalg.norm(chi,2),2) -1.0) );

	return mu_dot, mu_dot*chi;

## ~~~~~~~~~~~~~ Eigenvalue Computations ~~~~~~~~~~~~~~~~~~~
def EigenVals(A):

	EIGS = np.linalg.eigvals(A); # Modify this to reduce the calculation effort ??
	idx = EIGS.argsort()[::-1];
	eigenValues = EIGS[idx];
	print "EigenVals ", eigenValues[0:10], "\n" ## Comment Out if possible
	
	return eigenValues[0:5];

'''
## ~~~~~~~~~~~~~ Two_Param_funcs ~~~~~~~~~~~~~~~~~~~
def Netwon_It_2Param(Pr,sigma,R,N_modes,Re_1,Re_2,Y):

	L,L_inv,FF,N1,N2 = Matrices_Psuedo(Pr,sigma,R,N_modes,Re_1,Re_2,Y)
	X = Y[0:-1];

	# 1) Determine Steady State
	it, err = 0,1.0; 
	MAX_ITERS = 10;
	while (err > (1e-10)):	

		# a) Eval matrices
		if it > 0:
			N1 = N_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
			N2 = N2_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
		
		# b) Precondition terms
		F_X = np.matmul(L_inv, np.matmul(L+ N1,X) + FF); 
		DF_X = np.matmul(L_inv,L+N1+N2);

		# c) Solve for new soln
		X_new = X - np.linalg.solve(DF_X,F_X);

		# d) Compute Error	
		err = np.linalg.norm((X_new - X),2)/np.linalg.norm(X_new,2)							
		print 'Error ',err
		print 'Iteration: ', it,"\n"

		X = X_new;
		it+=1;
		if (it > MAX_ITERS):
			break

	Y[0:-1] = X;

	# 2) Determine Eigs	
	AA = np.matmul(M_inv,L+N1+N2);
	Eig1 = EigenVals(AA)[0:5];

	return Y, it, Eig1;

def Soln_Interp()
	
	for pp in xrange(len(X)):
		x = np.zeros(3); y = np.zeros(3)
		for jj in xrange(3):
			y[jj] = YvRa[-(jj+1)][pp,0];
			x[jj] = YvRa[-(jj+1)][-1,0];
		a = np.polyfit(x,y,2)
		Y[pp,0] = np.polyval(a,Y[-1,0]);
'''
# ~~~~~~~~~ Evolution G, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0], [X,mu] ~~~~~~~~~~~~~~~~~~~ #

#@profile
def G_Y(L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0): # Correct

	# 1) Unwrap Y
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; # Old solution for tangent

	args = [L,L_inv,FF,N1,N2,Y];

	# 2 ) Set Matrices  Correct
	ind = L.shape[0]; s=(ind+1,ind+1);
	GY = np.zeros(s);

	# ~~~ Line1 ~~~~ based on args !!
	#F_x = Evo_FF_X(*args); GY[0:ind,0:ind] = F_x; 
	GY[0:ind,0:ind] = Evo_FF_X(*args); ### MOST EXPENSIVE CALL
	#F_mu = Evo_FF_mu(L_Ra,L_inv,Y); 
	GY[0:ind,ind] = Evo_FF_mu(L_Ra,L_inv,Y).reshape(ind); 

	# ~~~~ Line2 ~~~~ based on args_0 !!
	GY[ind,0:ind] = delta*np.transpose(dX_0);
	GY[ind,ind] = (1.0-delta)*dmu_0; 
	
	return GY;

#@profile
def G(L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0): # Correct
	
	# a) Extract X & mu
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; # Old solution for tangent
	
	# b) Package arguments
	args = [L,L_inv,FF,N1,N2,Y];
		
	# c) Solving
	
	# Solve F(X,mu) = 0
	X_n = Evo_FF(*args) ### MOST EXPENSIVE CALL
	
	# Solve for N(X,mu,s) = 0, tangent condition
	N_n = delta*np.dot(np.transpose(dX_0),XX-X_0)[0,0] + (1.0-delta)*dmu_0*(mu - mu_0) - ds;

	# 3) Repackage into GG shape Y
	GG = np.zeros(Y.shape);
	GG[0:-1] = X_n; GG[-1,0] = N_n;
	
	return GG;

#@profile
def Pred_Y(Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D): # Correct
		
	# 1) Unwrap Y
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; 

	args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D];
	L,L_inv,FF,N1_0,N2_0 = Matrices_Psuedo(*args);
	args_0 = [L,L_inv,FF,N1_0,N2_0,Y_0];

	chi = xi(*args_0);
	dmu_dt, dx_dot =  D_dots(chi); #mu_dot(chi), X_dot(chi);

	'''det = np.linalg.det(np.matmul(L_inv,L+N1_0+N2_0))
	print "Determinant ",det

	if (det*prev_det < 0) and (abs(det) < 0.1):
		print "Bifurcation Change Determinant sign \n"
		#global sign
		#sign = -1*sign;
	global prev_det
	prev_det = det;

	print "dmu_0 ",dmu_dt
	print "||xi|| ",np.linalg.norm(dx_dot/dmu_dt,2),"\n"'''

	# 2) Predict & Update new guess
	Y_p = np.zeros(Y_0.shape);
	Y_p[0:-1] = X_0 + dx_dot*ds;
	Y_p[-1,0] = mu_0 + dmu_dt*ds; 

	return Y_p, dmu_dt, dx_dot;

#@profile
# Approx 40 times faster
def Pred_Y_Eff(GG_Y,Y_0):

	# 1) Unwrap Y + Create RHS
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1];
	B = np.zeros(Y_0.shape); B[-1,0] = 1.0;


	# 2) Calculate & Normalize
	Y_dot = np.linalg.solve(GG_Y,B);
	
	mu = Y_dot[-1,0];  X = Y_dot[0:-1];
	Y_dot = Y_dot/np.sqrt(delta*np.linalg.norm(X) + (1-delta)*np.linalg.norm(mu) );

	# 3) Unpack & Update new guess
	Y_p = np.zeros(Y_0.shape);
	dmu_dt, dx_dot = Y_dot[-1,0], Y_dot[0:-1];
	Y_p[0:-1] = X_0 + dx_dot*ds;
	Y_p[-1,0] = mu_0 + dmu_dt*ds;

	return Y_p, dmu_dt, dx_dot;

#@profile
def Correct_Y(Pr,sigma,D,R,N_modes,Re_1,Re_2,Y_0,Y,dmu_0,dX_0,c_eig): # Correct
	
	it = 0; err = 1.0
	# 1) Update Solution
	while err > 1e-10:
	
		# a) Previous old solution & current prediction
		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args);
		args1 = [L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0];

		# b) Update new solution using solve
		GG = G(*args1); GG_Y = G_Y(*args1);
		Y_new = Y - np.linalg.solve(GG_Y,GG);

		# Force psi_0, Omega_0 to 0
		#Y_new[len(R[1:-1]):3*len(R[1:-1]),0] = np.zeros(2*len(R[1:-1]));

		err = np.linalg.norm(Y_new - Y,2)/np.linalg.norm(Y_new,2);
		print "Iter ",it
		print "Error ",err
		print "Ra ",Y[-1,0],"\n"

		it+=1;
		Y = Y_new;

		# c) If it > 5 Repredict !!!!
		if it > 5:
			it = 0;
			global ds
			ds = ds/4.0; # Smaller for better cusp following
			print "Repredicting: reduced ds ",ds,"\n"
			
			#args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0];
			#L,L_inv,FF,N1_0,N2_0 = Matrices_Psuedo(*args);
			#args_0 = [L,L_inv,FF,N1_0,N2_0,Y_0];
			
			#chi = xi(*args_0); dmu_0, dX_0 =  D_dots(chi);
			Y[0:-1] = Y_0[0:-1] + dX_0*ds;
			Y[-1,0] = Y_0[-1,0] + dmu_0*ds;


	print "Total Iterations ",it,"\n"			
	# 2) Calculate Leading Eigenvectors
	#~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~
	if c_eig%eig_freq==0:
		A = np.matmul(M_inv,L+N1+N2);	
		eigenValues = EigenVals(A); ####
	else:
		eigenValues = [];
	#~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~
	
	# 3) Step Length Control		
	if it <= 2:
		#print ds
		global ds
		if ds < 300.0:
			ds = 2.0*ds;
			print "Increased ds ",ds,"\n"
		else:
			ds = 300.0	

	args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,Y];
	return args1,eigenValues,GG_Y;

#@profile
def Psuedo_Branch(Pr,sigma,D,R,N_modes,Re_1,Re_2,Y_0,Eig1):

	# 1) Arrays of steady states and Ra
	YvRa = []; LambdavRa = [];

	# Add initial Eig1 and steady state
	YvRa.append(Y_0); LambdavRa.append(Eig1)

	eps_curr = (Y_0[-1,0]-Ra_c)/Ra_c; ii = 0;
	Eig_Sign = 1.0; diff = -1.0;
	while (ii < iters): #(Eig_Sign.real > 0.0): #(ii < iters) or 

		print "#~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~#"
		print "Branch point ",ii,"\n"
		
		# a) Pred Initial guess for [X(s),mu(s)] with step ds
		if ii < 1:	
			args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D];
			Y, dmu_dt, dx_dot = Pred_Y(*args);
		else:	
			Y, dmu_dt, dx_dot = Pred_Y_Eff(GG_Y,Y_0);
			
		# b) Correct
		args = [Pr,d,D,R,N_Modes,Re_1,Re_2,Y_0,Y, dmu_dt, dx_dot, ii];
		args1,Eigs,GG_Y = Correct_Y(*args);
		Y = args1[-1];

		# c) Append Results 
		YvRa.append(Y); 

		if Eigs != []:
			LambdavRa.append(Eigs)

		# e) Termination Condition
		'''
		if ii > eig_freq:
			#print LambdavRa[ii][0]
			#print LambdavRa[ii-1][0]
			Eig_Sign = LambdavRa[ii][0]*LambdavRa[ii-1][0];
			print "Eig_sign ",Eig_Sign.real
			print "Ra ",Y[-1,0]
		'''
		# e) Update
		Y_0 = Y;
		eps_curr = (Y[-1,0]-Ra_c)/Ra_c;
		#print "Epsilon current ",eps_curr
		ii+=1

		np.save(STR1,YvRa)
		np.save(STR2,LambdavRa)

	return YvRa,LambdavRa;

def Nu(X):
	npr = len(R)-2
	dr = R[1]-R[0]; A_T = -(1.0+d)/d

	psi = np.zeros(N_Modes*npr)

	for k in xrange(N_Modes):
		ind = k*3*(len(R)-2)
		nr = len(R)-2;
		psi[k*nr:(k+1)*nr] = X[ind:ind+nr,0].reshape(nr);

	return np.linalg.norm(psi,2);

from mpmath import *
from sympy import *
import scipy.special as sp
def Temp_Prof(Y):

	N = len(Y); npr = len(R)-2; TT = np.zeros(len(R))

	A_T = -(1.0+d)/d

	ii = -1; #N-1;
	#Ra[ii] = Y[ii][-1,0];#(Y[ii][-1,0] -Ra_c)/Ra_c
	
	# T_0(r)
	TT[1:-1] = Y[ii][npr:2*npr,0];
	
	# 1) ~~~~~~~~ horizontally averaged temp profile ~~~~~~~~~~
	plt.figure(figsize=(12,8))
	
	# Lowest Ra
	TT[1:-1] = Y[0][npr:2*npr,0];
	xx = np.linspace(R[-1],R[0],100);
	T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT,'ko',markersize=3.0)
	plt.plot(xx,T,'k-',linewidth=1.5)

	# Highest Ra
	TT = np.zeros(len(R))
	TT[1:-1] = Y[ii][npr:2*npr,0];
	xx = np.linspace(R[-1],R[0],100);
	T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT,'bo',markersize=3.0) 
	plt.plot(xx,T,'b-',linewidth=1.5)


	plt.xlabel(r'$r$',fontsize=20);
	plt.ylabel(r'$\overline{T(r,\theta)}$',fontsize=20);
	plt.xlim([1.0,1.0+d])
	plt.show()


	# 2) ~~~~~~~~ Local to Poles theta = 1,89 ~~~~~~~~~~
	
	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 300; theta = np.linspace(f,(np.pi/2.0)-f,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	print "Pole ",theta[1]*(180.0/np.pi)
	print "Equator ",theta[-1]*(180.0/np.pi)
	
	T_p = np.zeros(len(R))
	T_e = np.zeros(len(R))

	# T(r,\theta = 1)
	for i in xrange(N_Modes):
		ind = i*3*npr
		T_p[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[1],i));
		T_e[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[-1],i));
	
	
	plt.figure(figsize=(12,8))
	
	# Pole
	T = np.polyval(np.polyfit(R,T_p,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_p = T_p + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'k-',linewidth=1.5,label=r'$\theta \approx 0$')
	plt.plot(R,T_p,'ko',markersize=3.0)

	# Equator
	T = np.polyval(np.polyfit(R,T_e,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_e = T_e + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'b-',linewidth=1.5,label=r'$\theta \approx \pi/2$')
	plt.plot(R,T_e,'bo',markersize=3.0)


	plt.xlabel(r'$r$',fontsize=20);
	plt.ylabel(r'$T(r,\theta)$',fontsize=20);
	plt.legend()
	plt.xlim([1.0,1.0+d])
	plt.show()

	# 3) ~~~~~~~~ Polar temp profile ~~~~~~~~~~

	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 100; theta = np.linspace(f,np.pi,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	T_0 = (-A_T/R[npr/2]) - (1.0/d);
	T_m = T_0*np.ones(len(theta))

	for i in xrange(N_Modes):
		ind = i*3*npr
		tt = Y[ii][ind+npr+npr/2,0]
		for jj in xrange(len(theta)):
			T_m[jj] += tt*float(P(theta[jj],i));
	
	
	plt.figure(figsize=(12,8))
	
	# Mid point r = 1 + 0.5*d
	plt.plot(theta,T_m,'k-',linewidth=1.5) #,label=r'$\theta \approx 0$')
	#plt.plot(theta,T_m,'ko',markersize=3.0)
	'''
	# Equator
	T = np.polyval(np.polyfit(R,T_e,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_e = T_e + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'b-',linewidth=1.5,label=r'$\theta \approx \pi/2$')
	plt.plot(R,T_e,'bo',markersize=3.0)
	'''

	plt.xlabel(r'$\theta$',fontsize=20);
	plt.ylabel(r'$T(r = 1 + d/2,\theta)$',fontsize=20);
	plt.legend()
	plt.xlim([0,np.pi])
	plt.show()

def Norm_v_Ra(Y):

	N = len(Y)
	X = np.zeros(N); Ra = np.zeros(N);
	npr = len(R)-2;
	TT = np.zeros(len(R))
	#dr = R[0]-R[1]; 
	A_T = -(1.0+d)/d
	for ii in xrange(N):
		X[ii] = Nu(Y[ii][0:-1])
		Ra[ii] = Y[ii][-1,0]/Ra_c; #(Y[ii][-1,0] -Ra_c)/Ra_c
		
		# Perhaps use Nu -1
		TT[1:-1] = Y[ii][npr:2*npr,0];
		
		''''
		# horizontally averaged temp profile
		if ii == 99:
			plt.figure(figsize=(12,8))
			TT[1:-1] = Y[0][npr:2*npr,0];
			xx = np.linspace(R[-1],R[0],100);
			T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
			TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
			plt.plot(R,TT,'ko',markersize=3.0)
			plt.plot(xx,T,'k-',linewidth=1.5)

			TT = np.zeros(len(R))
			TT[1:-1] = Y[ii][npr:2*npr,0];
			xx = np.linspace(R[-1],R[0],100);
			T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
			TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
			plt.plot(R,TT,'bo',markersize=3.0) 
			plt.plot(xx,T,'b-',linewidth=1.5)

			#plt.plot(R,((R**2)/A_T)*np.matmul(D,TT),'k--')
			plt.xlabel(r'$r$',fontsize=20);
			plt.ylabel(r'$\overline{T(r,\theta)}$',fontsize=20);
			plt.xlim([1.0,4.0])
			plt.show()
		'''

		NuM1 = ((R**2)/A_T)*np.matmul(D,TT)

		Bot = NuM1[0]

		Top = NuM1[-1]

		print Bot - Top

		X[ii] = 1 + Bot

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$Ra/Ra_c$',fontsize=20);
	plt.ylabel(r'$\log(Nu)$',fontsize=20);
	print "Exponnet ",np.log10(X[50:-1])/np.log10(1.96*Ra[50:-1]);
	plt.plot(Ra,np.log10(X),'k-',linewidth=1.5);
	plt.plot(Ra,np.log10(X),'y*',linewidth=0.5);
	plt.plot(Ra,np.log10( 1.96*(Ra**0.25 ) ) -0.2*np.ones(len(Ra)) ,'b.')
	plt.ylim([0,0.5])
	plt.xlim([0,30])
	plt.show()

def Lam_v_Ra(LAM,Y):

	N = len(Y);	Ra = np.zeros(N);
	LAM_Re = np.zeros((5,N)); LAM_IM = np.zeros((5,N));

	for ii in xrange(N):
		LAM_Re[:,ii] = LAM[ii][0:5].real;
		LAM_IM[:,ii] = LAM[ii][0:5].imag;
		print "Iteration ",ii
		print "Ra ",Y[ii][-1,0]
		print LAM[ii],"\n"
		
		Ra[ii] = Y[ii][-1,0];# -Ra_c)/Ra_c

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Re{\lambda}$',fontsize=20);
	line = ['r.','g*','bs','y^','ko']
	for ii in xrange(5):
		#ii = 0;
		plt.plot(Ra,LAM_Re[ii,:],line[ii],linewidth=1.5);
	plt.show()	

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Im{\lambda}$',fontsize=20);
	for ii in xrange(5):
		plt.plot(Ra,LAM_IM[ii,:],line[ii],linewidth=1.5);
	plt.show()


# Test Parameters to pass
l = 2; d = 3.0;
Ra_c =  409.5; #/(d**3);
Ra = Ra_c*(1+0.1)
Tau = 10.0; Pr = 1.0;
# Known Steady State
'''

# Test 2 Parameters
l = 10; d = 0.4;
Ra_c = 13000.0/(d**3);
eps = 0.1
Ra = Ra_c*(1+eps)
Tau = 1.0/15.0; Pr = 1.0;


# Full Parameters
l = 20; d = 0.2;
Ra_c = 1014578.0;
eps = 0.3
Ra = Ra_c*(1+eps)
Tau = 1.0/15.0; Pr = 1.0;
'''

Nr = 20; N_Modes = 30+1; # Must recompile other Fortran codes when this is altered!!

# Define all subsequent radial dimensions from R !!!
D,R = cheb_radial(Nr,d)

DIM = 3*len(R[1:-1])*N_Modes;
Y_0 = np.zeros((DIM+1,1))
X = np.zeros((DIM,1)) 
#X[:,0] = 0.0001*np.random.randn(DIM)
print "Dim ",DIM

#sys.exit()

# Load Initial Condition
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ Steady-State ~~~~~~~~~~~~~~~~~~~~~

# All Nr = 20, N_modes = 30 +1
Y = np.load("Test1_l2_E05_Y.npy");
X = Y[0:-1]; Ra = Y[-1,0];

# All Nr = 20, N_modes = 100 +1
#Y = np.load("Test_l10_E0.4_Y.npy");
#X = Y[0:-1]; Ra = Y[-1,0];

# 1) Time-stepping, can resize Polar N_modes 
dt = 1e-02; # As Cedric used 10^-5
RUNS = 100; #4*1000; # Approximately 1hr
X,X_VID = IMPLICIT(RUNS,Tau,Pr,Ra,d,Nr,N_Modes,dt,X)
Y_0[0:-1] = X; Y_0[-1,0] = Ra;
#sys.exit()
'''
#STR = "".join(['TestB_l10_E',str(eps),'_Y.npy']) 
#np.save(STR,Y_0)
#STR = "".join(['TestB_l10_E',str(eps),'_XVID.npy']) 
#np.save(STR,X_VID)

os.chdir("L10_Results")
STR = "".join(['TestB_l10_E',str(eps),'_Y.npy']) 
Y_0 = np.load(STR)
STR = "".join(['TestB_l10_E',str(eps),'_XVID.npy']) 
X_VID = np.load(STR)
'''

X_norm = [];
print X_VID.shape
for ii in range(len(X_VID[1,:])):
	#print X.shape
	X_norm.append(np.linalg.norm(X_VID[:,ii]));

Time = np.arange(0,len(X_norm),1)*dt
plt.figure(figsize=(10,8))
plt.plot(Time,X_norm,'k-',linewidth=1.5)
plt.xlabel(r"time t",fontsize=20)
plt.ylabel(r"$||X||$",fontsize=20)
plt.xlim([min(Time),max(Time)])
#plt.ylim([0,20])
plt.grid()
plt.show()


# 2) Steady State, can rezise Radial collocation
#D_o,R_o = cheb_radial(20,d); #XX = INTERP_SPEC(R,R_o,Y[0:-1]);
#Y_0 = np.zeros((DIM+1,1))
##Y_0[0:-1] = Netwon_It(Tau,Pr,Ra,d,Nr,N_Modes,X)
##Y_0[-1,0] = Ra
#np.save("Test1_l10_E01_Y.npy",Y_0)

'''
# 3) EigenVals
args = [Tau,Pr,Ra,d,D,R,N_Modes,Y_0];
M_inv,L,N1,N2 = Matrices_Psuedo(*args);
A = M_inv.dot(L+N1+N2)
Eig_0 = EigenVals(A)[0:5]
'''

# 4) Continuation


# 5) Plotting
from Projection import *
X = Y_0[0:-1]; 
#X[:,0] = X_VID[:,2]
N_Modes = 30+1 
f = 1e-11; Nth = 300; 
theta = np.linspace(f,np.pi-f,Nth); xx = np.linspace(R[-1],R[0],50);
#PSI, T, C, T_0 = Outer_Prod_PTC(R,theta,N_Modes,X,d)
PSI, T, C, T_0 = SPEC_TO_GRID(R,xx,theta,N_Modes,X,d)
Plot_Package_CHE(xx,theta,C,PSI,T,d)
N_Modes =40+1
Energy(X,N_Modes,R);
#sys.exit()


# 6) Make the video
#VID(R,theta,N_Modes,X_VID,d,Ra,dt)
