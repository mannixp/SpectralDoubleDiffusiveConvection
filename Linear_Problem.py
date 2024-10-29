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

def Eig_Vals(Ra1,l,d,Nvals, Ra_s=300,Pr=1,Tau=1./30., Nr = 30):	
	
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

def Eig_Vec( Ra1,l,d,    k, Ra_s=300,Pr=1,Tau=1./30., Nr = 30):
	
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

	return None

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

	l_left  =l_org - 2
	l_right =l_org + 2


	def Neutrals_Ra_D(Ra_org,l, d_org, k, width=0.05,N_iters = 30):	
		
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

	L = np.arange(l_org-3,l_org+3,1);

	# 1 Generate a figure
	fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(16,8),dpi=200,layout="constrained")

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


	Steady_bifurcation = 1
	for l in L:
		
		Ra_l,d_l = Neutrals_Ra_D(Ra_c_steady,l,d_org,Steady_bifurcation)

		if (l == l_left):
			Ra_left = Ra_l
		elif (l == l_right):
			Ra_right = Ra_l

		if l%2 == 0:
			if l == l_org:
				ax1.plot(Γ(d_l),Ra_l,'b:',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax1.plot(Γ(d_l),Ra_l,'k:',linewidth=2.0,label = r'$\ell =%d$'%l)
		else: 	
			if l == l_org+1:
				ax1.plot(Γ(d_l),Ra_l,'r-',linewidth=2.0,label = r'$\ell =%d$'%l)
			else:
				ax1.plot(Γ(d_l),Ra_l,'k-',linewidth=2.0,label = r'$\ell =%d$'%l)

	
	
	index = np.argmin(abs(Ra_left-Ra_right))
	d_mid = d_l[index]
	print('l=%d,%d intersection, d_mid = %3.5f, Γ(d_mid) = %3.3f'%(l_left,l_right,d_mid,Γ(d_mid)),'\n')

	# ax2.set_title(r'$\lambda = \pm i \omega$',fontsize=25);
	# ax2.set_xlabel(r'$\Gamma$',fontsize=30)
	# ax2.set_ylabel(r'$Ra$',fontsize=30)

	# ax2.grid()
	# ax2.legend(loc=2,fontsize=20)
	# ax2.tick_params(axis="both", labelsize=25,length=2,width=2)

	# ax1.set_title(r'$\lambda = 0$', fontsize=25)
	# #ax1.set_xlabel(r'$\Gamma$',fontsize=20)
	# ax1.set_ylabel(r'$Ra$',fontsize=30)

	# ax1.grid()
	# ax1.legend(loc=2,fontsize=20)
	# ax1.tick_params(axis="both", labelsize=25,length=2,width=2)

	# ax2.set_ylim([2800,3200])
	# ax1.set_ylim([9700,10000])

	plt.savefig("NeutralCurves_TauI15_Pr1_Ras500.png", format='png', dpi=200)
	plt.show()

	return None

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

	# ~~~~~# Validation Case (2) # ~~~~~#
	# Vary Tau
	d = 2; l = 2
	Ra_c = 6900; Ra_s = 500; Pr = 1; Tau = 10;
	Eig_val = Eig_Vals(Ra_c,l,d,Nvals=2,Ra_s=500,Pr=1,Tau=10, Nr = 20);
	print(Eig_val)

	# Vary Tau
	Ra_c = 7800; Ra_s = 500; Pr = 1; Tau = .5
	Eig_val = Eig_Vals(Ra_c,l,d,Nvals=2,Ra_s=500,Pr=1,Tau=.5, Nr = 20);
	print(Eig_val)

	# Vary Ra_s
	Ra_c = 7100; Ra_s=250; Pr=1; Tau=1;
	Eig_val = Eig_Vals(Ra_c,l,d,Nvals=2,Ra_s=250,Pr=1,Tau=1, Nr = 20);
	print(Eig_val)

	# ~~~~~# L = 11 Gap #~~~~~~~~~#
	# d = 0.31325; l=11.0; 
	# Ra_c_steady = 9775.905436191546
	# Ra_c_hopf   = 2879.0503253066827

	# ~~~~~# L = 10 Gap #~~~~~~~~~#
	#d = 0.3521; l=10.0; 
	#Ra_c_steady = 9851.537357677651; Nvals=1
	#Ra_c_hopf   = 2965.1798389922933; Nvals=0

	# # ~~~~~# L = 12 Gap #~~~~~~~~~#
	# d = 0.10778
	# l = 30 
	# Ra_c_steady = 9721.960152818841; Nvals=1
	# Ra_c_hopf   = 2965.1798389922933; Nvals=0
	
	# Ra = Critical_Eigval(Ra_c_steady,l,d,Nvals=1)
	# #Ra = Critical_Eigval(Ra_c_hopf,l,d,Nvals=0)

	# print('Ra = ',Ra)
	# #d = 0.3341; # Half of the two avbove
	# #Neutral(Ra_c_hopf,Ra_c_steady,l,d_org=d)

	# # Compute the eigenvector
	# Nr = 20
	# lambda_i = 1 # For steady-bifurcation
	# #lambda_i = 0 # For Hopf-bifurcation
	# Eig_val = Eig_Vals(Ra,l,d,Nvals = 2 ,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	# Eig_vec = Eig_Vec( Ra,l,d,k=lambda_i,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	
	# print('\n #~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~#')
	# print('Eigen Values = ',Eig_val)
	# print('Chose Eigenvector for \lambda_%d = %e'%(lambda_i,Eig_val[lambda_i]) )
	# print('#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~# \n')

	# filename = 'EigVec_l30.npy'
	# N_fm = 64
	# X    = Full_Eig_Vec(Eig_vec,l,N_fm,nr=Nr-1,symmetric=False)
	# np.save(filename,X)

	# from Plot_Tools import Cartesian_Plot,Energy
	# Energy(filename,frame=-1)
	# Cartesian_Plot(filename,frame=-1,Include_Base_State=False)

	return None;



def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:

	def __init__(self, denominator=4, number=np.pi, latex='\pi'):
		self.denominator = denominator
		self.number = number
		self.latex = latex
	
	def locator(self):
		return plt.MultipleLocator(self.number / self.denominator)
	
	def formatter(self):
		return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def velocity_field(X_hat, D, R,xx, N_fm):
	"""
	Compute the velocity vector 

	u_r = 1/(r^2 sin θ) ∂(ψsinθ)/∂θ = (1/r^2) J_θ(ψ) ~ cosine

	u_θ =  -(1/r) ∂ψ/∂r 			=  -(1/r) D_r(ψ) ~ sine

	"""

	from Matrix_Operators import J_theta_RT
	from Transforms import IDCT,IDST,grid
	from scipy.interpolate import interp1d

	nr = len(R[1:-1])
	N  = N_fm*nr;
	Dr = D[1:-1,1:-1];
	IR = np.diag(1./R[1:-1]);
	IR2 = np.diag(1./R[1:-1]**2);

	ψ_hat=X_hat[0:N]
	JPSI =J_theta_RT(ψ_hat, nr,N_fm)

	Jψ_hat = np.zeros((nr, N_fm)) 	
	Dψ_hat = np.zeros((nr, N_fm))
	for k in range(N_fm):
		ψ_k          = ψ_hat[k*nr:(1+k)*nr];
		Dψ_hat[:,k]  = Dr@ψ_k                 # ~ sine
		Jψ_hat[:,k]  = JPSI[k*nr:(1+k)*nr] # ~ cosine
	
	# Convert Sine to sinusoids
	Dψ_hat[:,1:] = Dψ_hat[:,0:-1]; Dψ_hat[:,0] = 0.0;

	# A) Assign velocity
	u_r = np.zeros((len(R),N_fm)); u_r[1:-1,:] = IDCT(IR2@Jψ_hat,n = N_fm) 
	u_θ = np.zeros((len(R),N_fm)); u_θ[1:-1,:] = IDST(-IR@Dψ_hat,n = N_fm)

	# B) Visualisation grid
	fu_r = interp1d(R, u_r, axis=0)
	fu_θ = interp1d(R, u_θ, axis=0)
	
	return fu_r(xx), fu_θ(xx)

def figure_1():

	# ~~~~~# L = 11 Gap #~~~~~~~~~#
	l = 11 
	d = 0.31325 
	Ra_c_steady = 9775.905436191546
	Ra_c_hopf = 2879.0503253066827

	# ~~~~~# L = 10 Gap #~~~~~~~~~#
	l = 10 
	d += 0.3521 
	Ra_c_steady += 9851.537357677651
	Ra_c_hopf += 2965.1798389922933
	
	# Create figure 1
	Neutral(Ra_c_hopf/2,Ra_c_steady/2,l,d_org=d/2)

	return None

def figure_2():
	
	Nr = 20
	N_fm = 64
	RES = 20

	from Matrix_Operators import cheb_radial
	from Plot_Tools import Spectral_To_Gridpoints

	# ~~~~~# L = 10 Gap #~~~~~~~~~#
	d = 0.3521
	l = 10 
	Ra_c_steady = 9851.537357677651
	Ra_c_hopf   = 2965.1798389922933
	
	lambda_i = 1 # For steady-bifurcation
	Eig_val = Eig_Vals(Ra_c_steady,l,d,Nvals = 2 ,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	Eig_vec = Eig_Vec( Ra_c_steady,l,d,k=lambda_i,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	
	print('\n #~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~#')
	print('Eigen Values = ',Eig_val)
	print('Chose Eigenvector for \lambda_%d = %e'%(lambda_i,Eig_val[lambda_i]) )
	print('#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~# \n')

	X = Full_Eig_Vec(Eig_vec,l,N_fm,nr=Nr-1,symmetric=False)

	D,R = cheb_radial(Nr, d) 
	Theta_grid = np.linspace(0,np.pi,N_fm) 
	r_grid     = np.linspace(R[0],R[-1],50)
	
	PSI, T, S, T_0 = Spectral_To_Gridpoints(X,R,r_grid,N_fm,d)
	
	# 1) Fix \theta labels to be [0,pi]
	fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24,8), dpi=200, layout='constrained')

	x_0 = np.linspace(0,np.pi,N_fm) 
	x_1 = np.linspace(R[0],R[-1],50)
	U = velocity_field(X,D,R,x_1,N_fm)
	ax[0,0].streamplot(x_0, x_1, U[1], U[0], density=.75,arrowsize=2)

	ax[1,0].contour( Theta_grid,r_grid,T,  RES, colors = 'k', linewidths=0.5,);
	ax[1,0].contourf(Theta_grid,r_grid,T,  RES, cmap="RdBu_r")
		
	ax[2,0].contour( Theta_grid,r_grid,S,  RES, colors = 'k',linewidths=0.5);
	ax[2,0].contourf(Theta_grid,r_grid,S,  RES, cmap="RdBu_r")
	
	# ~~~~~# L = 11 Gap #~~~~~~~~~#
	d = 0.31325 
	l = 11 
	Ra_c_steady = 9775.905436191546
	Ra_c_hopf   = 2879.0503253066827

	lambda_i = 1 # For steady-bifurcation
	Eig_val = Eig_Vals(Ra_c_steady,l,d,Nvals = 2 ,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	Eig_vec = Eig_Vec( Ra_c_steady,l,d,k=lambda_i,Ra_s=500.0,Pr=1.0,Tau=1./15.,Nr=Nr)
	
	print('\n #~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~#')
	print('Eigen Values = ',Eig_val)
	print('Chose Eigenvector for \lambda_%d = %e'%(lambda_i,Eig_val[lambda_i]) )
	print('#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~# \n')

	X = Full_Eig_Vec(Eig_vec,l,N_fm,nr=Nr-1,symmetric=False)

	D, R = cheb_radial(Nr, d)
	Theta_grid = np.linspace(0,np.pi,N_fm) 
	r_grid     = np.linspace(R[0],R[-1],50)

	PSI, T, S, T_0 = Spectral_To_Gridpoints(X,R,r_grid,N_fm,d)

	x_0 = np.linspace(0,np.pi,N_fm) 
	x_1 = np.linspace(R[0],R[-1],50)
	U = velocity_field(X,D,R,x_1,N_fm)
	ax[0,1].streamplot(x_0, x_1, U[1], U[0], density=.75,arrowsize=2)

	ax[1,1].contour( Theta_grid,r_grid,T,  RES, colors = 'k', linewidths=0.5,);
	ax[1,1].contourf(Theta_grid,r_grid,T,  RES, cmap="RdBu_r")

	ax[2,1].contour( Theta_grid,r_grid,S,  RES, colors = 'k',linewidths=0.5);
	ax[2,1].contourf(Theta_grid,r_grid,S,  RES, cmap="RdBu_r")

	for i in range(2):
		ax[0,i].set_title(r'$(u_r,u_{\theta})$',fontsize=30)
		ax[1,i].set_title(r'$\Theta$',fontsize=30)
		ax[2,i].set_title(r'$\Sigma$',fontsize=30)
		ax[2,i].set_xlabel(r'$\theta$',fontsize=30)

		for j in range(3):

			if i == 0:
				ax[j,i].set_ylabel(r'$r$',fontsize=30)


			ax[j,i].tick_params(axis="y", labelsize=25,length=2,width=2)

			if j == 2:
				ax[j,i].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
				ax[j,i].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
				ax[j,i].tick_params(axis="x", labelsize=25,length=2,width=2)
			else:
				ax[j,i].set_xticks([])

	plt.savefig("Eig_Vec.png",format='png', dpi=200)
	plt.show()

	return None

def large_aspect_neutrals():

	# ~~~~~# L = 30 Gap #~~~~~~~~~#
	l = 10
	d = np.pi/9 
	Ra_c_steady = 9851.537357677651
	Ra_c_hopf = 2965.1798389922933
	
	# Create figure 1
	Neutral(Ra_c_hopf,Ra_c_steady,l,d_org=d)

	return None

# Execute main
if __name__ == "__main__":

	# %%
	#%matplotlib inline

	# %%
	#figure_1()
	#figure_2()
	large_aspect_neutrals()

	# %%
	#main_program();

# %%
