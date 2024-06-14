#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from Linear_Matrix_Operators import ML_0, Ll_0, cheb_radial

Γ = lambda d,R_1 = 1: (R_1*np.pi)/d

def Eig_Vals(Ra1,l,d,Nvals, Ra_s=500,Pr=1,Tau=1./15., Nr = 30):	
	
	"""
	Solve the EVP for a given 

	Inputs:
	Ra - float Rayleigh Number
	l  - float legendre polynomial mode number
	d  - float gap width 

	Nvals - integer
	
	# 0 for Hopf-bifurcation
	# 1 for the first steady bifurcation
	# (2-10) for the first N eigenvalues
	
	Returns:
	
	Eigenvalues - float
	
	or 

	Eigenvalues - complex array

	"""

	D,R = cheb_radial(Nr,d);

	M = ML_0(D,R,l); A = Ll_0(D,R,d,l,Ra1,Ra_s,Pr,Tau);
	AA = np.matmul(np.linalg.inv(M),A)
	eigenValues = np.linalg.eigvals(AA)

	# Sort eigenvalues
	idx = eigenValues.real.argsort()[::-1]   

	if (Nvals == 0) or (Nvals == 1):
		return eigenValues[idx][Nvals].real;
	else:
		return eigenValues[idx][0:Nvals];	

def Eig_Vec( Ra1,l,d,    k, Ra_s=500,Pr=1,Tau=1./15., Nr = 30):
	
	"""
	Solve the EVP for a given 

	Inputs:
	Ra - float Rayleigh Number
	l  - float legendre polynomial mode number
	d  - float gap width 

	k - integer selects the k'th eigenvector
	
	Returns:

	Eigenvector - complex array

	"""

	D,R = cheb_radial(Nr,d);

	M = ML_0(D,R,l); A = Ll_0(D,R,d,l,Ra1,Ra_s,Pr,Tau);
	AA = np.matmul(np.linalg.inv(M),A)
	eigenValues,eigenVectors = np.linalg.eig(AA)

	# Sort eigenvalues
	idx = eigenValues.real.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]

	return eigenVectors[:,k].real

def Critical_Eigval(Ra1,l,d, Nvals=1):

	"""
	Given an initial guess for the critical Eigenvalue local this point
	i.e. Newton solve for lambda.real = 0

	Inputs:
	Ra - float Rayleigh Number
	l  - float legendre polynomial mode number
	d  - float gap width 

	Nvals - integer
	
	# 0 for Hopf-bifurcation
	# 1 for the first steady bifurcation
	# (2-10) for the first N eigenvalues
	
	Returns:
	
	Ra_c - float critical Rayleigh number

	"""

	import scipy.optimize as scp

	Ra_c = scp.newton( Eig_Vals, x0 = Ra1, args = (l,d,Nvals),tol = 1e-05, maxiter = 30).real;
	
	#print("l = %d, Ra_c = %e \n "%(l,Ra_c) )

	return Ra_c;

def Ra_Stability_Trace(Ra_c,d,Nvals):

	"""
	Scan a range of Ra for a given set of parameters and plot 

	the leading eigenvalue(s) vs. the control parameter

	Input:
	Ra_c - float critical Rayleigh number/ starting point
	d    - float gap width
	Nvals - integer see eigvals definition for details
	
	Returns:

	None;

	"""

	fig, (ax1, ax2) = plt.subplots(nrows=2)
	
	N = 50; # Scan_resolution)

	eps = np.linspace(-0.75,0.02,N);

	for l in range(8,13,2):

		print('l=',l)
		EIG = np.zeros((N,Nvals),dtype=np.complex_)
		
		for ii in range(N):
			Ra1 = Ra_c*(1.0+eps[ii]);
			EIG[ii,:] = Eig_Vals(Ra1,l,d,Nvals);
		
		for k in range(Nvals):
			ax1.plot(eps,EIG[:,k].real,linewidth=2,label='l=%d'%l);
		
		# plt imag	
		for k in range(Nvals):
			ax2.plot(eps,EIG[:,k].imag,linewidth=2,label='l=%d'%l);

	ax1.plot(eps,0.0*eps,'k--')
	ax1.legend(fontsize=20)
	ax2.legend(fontsize=20)
	ax2.set_xlabel(r'$(Ra - Ra_c)/Ra$',fontsize=26)
	ax1.set_ylabel(r'$\Re(\lambda)$',fontsize=26);
	ax2.set_ylabel(r'$\Im(\lambda)$',fontsize=26);
	#ax1.set_ylim([-200,200])
	#ax1.set_xlim([-0.1,0.5])
	#ax2.set_xlim([-0.1,0.5])
	plt.tight_layout()
	plt.show()

	return None;

def Neutral(Ra_c_hopf,Ra_c_steady,l_org,d_org):
	
	"""
	Given an initial starting point (Ra,d) and mode number l, 
	generate a plot of the neutral stability curves in Ra,d space
	
	!!!! Note Currently this is set up for l=10 !!!

	Inputs:

	Ra_org  - float rayleigh number
	d_org   - float gap width
	l 		- float/integer mode number
	k       - integer 0 or 1 depending on steady bifurcation or Hopf-bifurcation

	Returns: 
	
	None;

	"""

	def Neutrals_Ra_D(Ra_org,l, d_org, k, width=0.05,N_iters = 200):	
		
		d_for  = np.linspace(d_org,d_org+width,N_iters);
		Ra_for = np.zeros(N_iters);

		Ra = Ra_org;
		for ii in range(N_iters):
			d 		   = d_for[ii];
			Ra_for[ii] = Critical_Eigval(Ra ,l,d,k)
			Ra         = Ra_for[ii];

		d_bck  = np.linspace(d_org,d_org-width,N_iters);
		Ra_bck = np.zeros(N_iters);

		Ra = Ra_org;
		for ii in range(N_iters):
			d 		   = d_bck[ii];
			Ra_bck[ii] = Critical_Eigval(Ra ,l,d,k)
			Ra         = Ra_bck[ii];	

		Ra_l = np.hstack((Ra_bck[::-1],Ra_for))
		d_l  = np.hstack(( d_bck[::-1], d_for))

		return Ra_l,d_l


	L = np.arange(l_org-2,l_org+4,1);

	# 1 Generate a figure
	fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(12,8),dpi=1200);

	ax2.set_title(r'$\lambda = \pm i \omega$',fontsize=20);

	Hopf_bifurcation = 0
	for l in L:
		
		Ra_l,d_l = Neutrals_Ra_D(Ra_c_hopf,l,d_org,Hopf_bifurcation)

		if l%2 == 0:
			if l == 10:
				ax2.plot(Γ(d_l),Ra_l,'b:',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax2.plot(Γ(d_l),Ra_l,'k:',linewidth=2.0,label = r'$\ell =%d$'%l)
		else: 	
			if l == 11:
				ax2.plot(Γ(d_l),Ra_l,'r-',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax2.plot(Γ(d_l),Ra_l,'k-',linewidth=2.0,label = r'$\ell =%d$'%l)

	ax2.set_xlabel(r'$\Gamma$',fontsize=25)
	ax2.set_ylabel(r'$Ra$',fontsize=25)

	ax2.grid()
	ax2.legend(loc=2,fontsize=20)

	ax2.tick_params(axis="x", labelsize=20,length=2,width=2)
	ax2.tick_params(axis="y", labelsize=20,length=2,width=2)

	ax1.set_title(r'$\lambda = 0$', fontsize=20)

	Steady_bifurcation = 1
	for l in L:
		
		Ra_l,d_l = Neutrals_Ra_D(Ra_c_steady,l,d_org,Steady_bifurcation)

		if (l == 8):
			Ra_8 = Ra_l
		if (l == 10):
			Ra_10 = Ra_l
		elif (l == 12):
			Ra_12 = Ra_l

		if l%2 == 0:
			if l == 10:
				ax1.plot(Γ(d_l),Ra_l,'b:',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax1.plot(Γ(d_l),Ra_l,'k:',linewidth=2.0,label = r'$\ell =%d$'%l)
		else: 	
			if l == 11:
				ax1.plot(Γ(d_l),Ra_l,'r-',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax1.plot(Γ(d_l),Ra_l,'k-',linewidth=2.0,label = r'$\ell =%d$'%l)

	#ax1.set_xlabel(r'$\Gamma$',fontsize=20)
	#ax1.set_xticks([])
	ax1.set_ylabel(r'$Ra$',fontsize=25)

	ax1.grid()
	ax1.legend(loc=2,fontsize=20)

	ax1.tick_params(axis="x", labelsize=20,length=2,width=2)
	ax1.tick_params(axis="y", labelsize=20,length=2,width=2)


	index = np.argmin(abs(Ra_12-Ra_8))
	d_mid = d_l[index]
	print('l=8,12 intersection, d_mid = %3.5f, Γ(d_mid) = %3.3f'%(d_mid,Γ(d_mid)),'\n')

	index = np.argmin(abs(Ra_10-Ra_12))
	d_mid = d_l[index]
	print('l=10,12 intersection, d_mid = %3.5f, Γ(d_mid) = %3.3f'%(d_mid,Γ(d_mid)),'\n')

	#l=10
	ax2.set_ylim([2800,3200])
	#ax2.set_xlim([0.325,0.375])

	#l=10
	ax1.set_ylim([9700,10000])
	#ax1.set_xlim([0.325,0.375])

	plt.tight_layout()
	plt.savefig("NeutralCurves_TauI15_Pr1_Ras500.png", format='png', dpi=200)
	plt.show();

	return None;

def Full_Eig_Vec(f,l,N_fm,nr,symmetric=False):

	from Transforms import grid,DCT,DST
	from Matrix_Operators import Vecs_to_X
	from scipy.special import eval_legendre, eval_gegenbauer

	# Divided by sin(θ) due to different definitions of the stream function
	θ  = grid(N_fm)
	#Gl = -(np.sin(θ)**2)*eval_gegenbauer(l-1,1.5,np.cos(θ))
	Gl = -np.sin(θ)*eval_gegenbauer(l-1,1.5,np.cos(θ))  
	Pl = eval_legendre(l,np.cos(θ))

	Gl_hat = DST(Gl,n=N_fm)#[0:l+1] 
	Pl_hat = DCT(Pl,n=N_fm)#[0:l+1] 

	# Convert from sinusoids back into my code's convention
	Gl_hat[0:-1] = Gl_hat[1:]; Gl_hat[-1] = 0.0;

	PSI = np.outer(f[0*nr:1*nr],Gl_hat)	
	T   = np.outer(f[1*nr:2*nr],Pl_hat)
	C   = np.outer(f[2*nr:3*nr],Pl_hat)

	return Vecs_to_X(PSI,T,C, N_fm,nr, symmetric)

def main_program():

	# # ~~~~~# Validation Case (1) # ~~~~~#
	#d = 2; Ra_c = 7268.365; l =2;
	#Eig_val = Eig_Vals(Ra_c,l,d,0,Ra_s=500,Pr=1.,Tau=1., Nr = 20);
	#print(Eig_val)

	# # ~~~~~# Validation Case (2) # ~~~~~#
	# d    = 2.0; Ra_c = 6.77*(10**3) - 10.; l=2;
	# Eig_val = Eig_Vals(Ra_c,l,d,0,Ra_s=0,Pr=1.,Tau=1., Nr = 30);
	# print(Eig_val)

	# ~~~~~# L = 11 Gap #~~~~~~~~~#
	# d = 0.31325; l=11.0; 
	# Ra_c_steady = 9775.905436191546
	# #Ra_c_hopf   = 2879.0503253066827
	# Ra = Critical_Eigval(Ra_c_steady,l,d,Nvals=1); 
	# print('Ra=',Ra)

	# ~~~~~# L = 10 Gap #~~~~~~~~~#
	d = 0.3521; l=10.0; 
	Ra_c_steady = 9851.537357677651
	#Ra_c_hopf   = 2965.1798389922933
	Ra = Critical_Eigval(Ra_c_steady,l,d,Nvals=1); 
	print('Ra=',Ra)

	# d = 0.3341; # Half of the two avbove
	# Neutral(Ra_c_hopf,Ra_c_steady,l,d_org=d)
		
	Nr = 20;
	lambda_i = 1
	Eig_val = Eig_Vals(Ra,l,d,Nvals = 2 ,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	Eig_vec = Eig_Vec( Ra,l,d,k=lambda_i,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	
	print('\n #~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~#')
	print('Eigen Values = ',Eig_val)
	print('Chose Eigenvector for \lambda_%d = %e'%(lambda_i,Eig_val[lambda_i]) )
	print('#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~# \n')

	filename = 'EigVec_l10.npy'
	N_fm = 64
	X    = Full_Eig_Vec(Eig_vec,l,N_fm,nr=Nr-1,symmetric=False)
	np.save(filename,X)

	from Plot_Tools import Cartesian_Plot,Energy
	Energy(filename,frame=-1)
	Cartesian_Plot(filename,frame=-1,Include_Base_State=False)

	return None;


# Execute main
if __name__ == "__main__":

	# %%
	%matplotlib inline

	# %%
	main_program();

# %%
