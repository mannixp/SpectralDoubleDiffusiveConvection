#! /usr/bin/env python

#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------

import numpy as np
#import pylab
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

import warnings
warnings.simplefilter('ignore', np.RankWarning)


from numpy.linalg import inv
import sys, time, os, math
from tempfile import TemporaryFile
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf) #Ensures Entire Matrix is spewed out
sys.setcheckinterval(500)

from mpmath import *
from sympy import *
import scipy.special as sp
#from Thermal_Anim import *

RES = 30;
# ------ To load from a previous initial Cond --- Using the $ell=2, m = 4$ mode
'''Stringy = ['/home/mannixp/Dropbox/Imperial/PhD_Thermal/Rotating/Base_States/Fine_ROT_IMP_SIGMA_1.50Ra_1777.8Re_5.00']
os.chdir("".join(Stringy))
P = np.load("Parameters.npy")

sigma = round(P[1],1)
print "Annulus Width ",sigma	
Nr = P[12]
print "Radial Steps", Nr 
Nth = P[13]
print "Azimuthal Steps ", Nth

R = np.linspace(1.0,1.0+sigma,Nr)
f = 1e-11
theta = np.linspace(f,np.pi-f,Nth)

# 2) Single frames of final state
OMEGA_NUM = np.load("OmegaSF.npy")
PSI_NUM = np.load("Stream_functionSF.npy")
T_NUM = np.load("ThermalSF.npy")
'''

# 300 x 100, N_Pts x N_modes
'''
GEG_k = np.zeros((300,101))
LEG_k = np.zeros((300,101))
#GEG_k1 = np.zeros((300,101))
#LEG_k1 = np.zeros((300,101))
## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
P = lambda x,kk: legendre(kk,cos(x))
f = 1e-11; Nth = 300; theta = np.linspace(f,np.pi-f,Nth);
for k in xrange(101):
	print "k ",k,"\n"
	for jj in xrange(Nth):
		GEG_k[jj,k] = G(theta[jj],k);
		LEG_k[jj,k] = P(theta[jj],k);
		#GEG_k1[jj,k] = G(theta[jj],k)/np.sin(theta[jj])
		#LEG_k1[jj,k] = P(theta[jj],k)*np.sin(theta[jj]);

np.save("GEG_k.npy",GEG_k);
np.save("LEG_k.npy",LEG_k);
'''
#np.save("GEG_k1.npy",GEG_k1);
#np.save("LEG_k1.npy",LEG_k1);


Geg_k = np.load("GEG_k.npy")
Leg_k = np.load("LEG_k.npy")
#Geg_k1 = np.load("GEG_k1.npy")
#Leg_k1 = np.load("LEG_k1.npy")

'''
theta = np.linspace(0.0,np.pi,300);
print theta
LINE = [':','-','-.','--'] 
for l in xrange(5):
	plt.plot(theta,Leg_k[:,l],linestyle=LINE[0],label=r'l%1.0f'%l)
	plt.plot(np.pi*np.ones(len(theta))+theta,Leg_k[::-1,l],linestyle=LINE[0]);
plt.legend()
plt.show()
for l in xrange(5):
	plt.plot(theta,Geg_k[:,l],linestyle=LINE[1],label=r'l%1.0f'%l)
	plt.plot(np.pi*np.ones(len(theta))+theta,Geg_k[::-1,l],linestyle=LINE[1]);
plt.legend()
plt.show()
sys.exit()
'''

def Inner_Prod(r,theta,k):

	Psi_k, T_k = np.zeros(len(theta)), np.zeros(len(theta)); # Obtained the required shape of numerical vector
	Psi_k1, T_k1 = np.zeros(len(theta)), np.zeros(len(theta)); # Obtained the required shape of numerical vector
	mp.dps, eps = 20, 1e-4;

	## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
	G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
	P = lambda x,kk: legendre(kk,cos(x))
	'''
	for jj in xrange(len(theta)):
		Psi_k[jj] = G(theta[jj],k); #/np.sin(theta[jj])
		T_k[jj] = P(theta[jj],k); #*np.sin(theta[jj]);
		Psi_k1[jj] = G(theta[jj],k)/np.sin(theta[jj])
		T_k1[jj] = P(theta[jj],k)*np.sin(theta[jj]);
	'''

	Psi_k = Geg_k[:,k]; #G(theta[jj],k); #/np.sin(theta[jj])
	T_k = Leg_k[:,k]; #P(theta[jj],k); #*np.sin(theta[jj]);
	Psi_k1 = Geg_k1[:,k]; #G(theta[jj],k)/np.sin(theta[jj])
	T_k1 = Leg_k1[:,k];  #P(theta[jj],k)*np.sin(theta[jj]);	



	N_psi = np.dot(Psi_k[:],Psi_k1[:]); # Include correct integral
	N_T = np.dot(T_k[:],T_k1[:]);

	#print "N_T ",N_T

	if N_psi == 0.0:
		Psi_k1 = np.zeros(len(theta));
	else:
		Psi_k1 = Psi_k1/N_psi;

	if N_T == 0.0:
		T_k1 = np.zeros(len(theta));
	else:
		T_k1 = T_k1/N_T;
	
	return Psi_k1, T_k1;

def Energy(X,N_Modes,R):
	plt.figure(figsize=(8, 6))

	E = np.zeros(N_Modes);
	for l in xrange(N_Modes):
		if l == 0:
			E[l] = np.linalg.norm(X[(len(R)-2):3*(len(R)-2),0],2);
		elif l == 1:
			E[l] = np.linalg.norm(X[3*(len(R)-2):6*(len(R)-2),0],2);
		else:
			ind = 6*(len(R)-2) + (l-2)*3*(len(R)-2);
			E[l] = np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);
	print "argmax(E) ",np.max(E),"\n"
	#E = np.log10(E[:]);
	#print E
	#E[np.isneginf(E)] = 0;
	#print E
	LL = np.arange(0,N_Modes,1)[:]
	#E = [];
	#for l in xrange(N_Modes):
	#	plt.semilogy(l,E[l], 'b:')

	plt.semilogy(LL[:],E[:], 'k:')
	plt.xlabel(r'Mode number $\ell$', fontsize=16)
	plt.ylabel(r'$log10( ||X_{\ell}|| )$', fontsize=16)
	plt.xticks(np.arange(0,N_Modes,10))
	plt.xlim([0,N_Modes-1])
	plt.grid()
	plt.show()

def Outer_Prod_PTC(R,theta,N_modes,X,d): # Fix this code to extend to the boundaries

	Psi_k, T_k, C_k = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R)); # Obtained the required shape of numerical vector
	T_0 = np.zeros(len(R));	T_00 = np.zeros(len(R));

	Gk, Pk = np.zeros(len(theta)), np.zeros(len(theta))
	mp.dps, eps = 20, 1e-4;

	s = (len(R),len(theta));
	PSI, T, C = np.zeros(s), np.zeros(s), np.zeros(s); 

	## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
	G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
	P = lambda x,kk: legendre(kk,cos(x))
	
	A_T, B_T = -(1.0+d)/d, -1.0/d;
	alpha = 1.0+d; #eta = Re_2/Re_1; 
	

	for ii in xrange(len(R)):
		T_0[ii] = -A_T/R[ii] + B_T;

	for k in xrange(N_modes):
		#print " k ",k,"\n"
		#for jj in xrange(len(theta)):
		#	Gk[jj] = G(theta[jj],k);
		#	Pk[jj] = P(theta[jj],k);
		Gk = Geg_k[:,k];
		Pk = Leg_k[:,k];

		ind = k*3*(len(R)-2)
		nr = len(R)-2;
		Psi_k[1:-1] = X[ind:ind+nr]; #.reshape(nr);
		T_k[1:-1] = X[ind+nr:ind+2*nr]; #.reshape(nr);
		C_k[1:-1] = X[ind+2*nr:ind+3*nr]; #.reshape(nr);

		'''print "PSI ",(np.max(Psi_k),np.min(Psi_k))
		print "T ",(np.max(T_k),np.min(T_k))
		print "Omega ",(np.max(Omega_k),np.min(Omega_k))
		'''
		if k==0:
			T_00 = np.outer(T_0,Pk)
		if k == 0:
			PSI = PSI + np.outer(Psi_k,Gk)
			T = T + np.outer(T_k,Pk) + np.outer(T_0,Pk); # Thermal Base State Added
			C = C +	np.outer(C_k,Pk) + np.outer(T_0,Pk);
		else:
			PSI = PSI + np.outer(Psi_k,Gk)
			T = T + np.outer(T_k,Pk)
			C= C +	np.outer(C_k,Pk)

	return PSI, T, C,T_00;

def SPEC_TO_GRID(R,xx,theta,N_modes,X,d): # Fix this code to extend to the boundaries

	Nr = len(R)
	Psi_k, T_k,C_k = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R)); # Obtained the required shape of numerical vector
	T_0 = np.zeros(len(R)); T_00 = np.zeros(len(R)); 
	
	Gk, Pk = np.zeros(len(theta)), np.zeros(len(theta))
	mp.dps, eps = 20, 1e-4;


	# Define New Radial Grid and Vectors
	s = (len(xx),len(theta));
	PSI, T, C = np.zeros(s), np.zeros(s), np.zeros(s); 

	## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
	G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
	P = lambda x,kk: legendre(kk,cos(x))
	
	A_T, B_T = -(1.0+d)/d, -1.0/d;
	alpha = 1.0+d; #eta = Re_2/Re_1; 
	

	for ii in xrange(len(R)):
		T_0[ii] = -A_T/R[ii] + B_T;

	# Interpolate	
	#Psi_0 = np.polyval(np.polyfit(R,Psi_0,Nr),xx)
	t_0 = np.polyval(np.polyfit(R,T_0,Nr),xx)
	#c_0 = np.polyval(np.polyfit(R,Omega_0,Nr),xx)

	print "X shape ",X.shape
	for k in xrange(N_modes):
		#print " k ",k,"\n"
		#for jj in xrange(len(theta)):
		#	Gk[jj] = G(theta[jj],k);
		#	Pk[jj] = P(theta[jj],k);
		Gk = Geg_k[:,k];
		Pk = Leg_k[:,k];

		nr = len(R)-2; ind = k*3*nr;

		Psi_k[1:-1] = X[ind:ind+nr,0]; #.reshape(nr);
		T_k[1:-1] = X[ind+nr:ind+2*nr,0]; #.reshape(nr);
		C_k[1:-1] = X[ind+2*nr:ind+3*nr,0]; #.reshape(nr);

		psi_k = np.polyval(np.polyfit(R,Psi_k,Nr),xx)
		t_k = np.polyval(np.polyfit(R,T_k,Nr),xx)
		c_k = np.polyval(np.polyfit(R,C_k,Nr),xx)

		if k==0:
			T_00 = np.outer(t_0,Pk)
		if k == 0:

			PSI = PSI + np.outer(psi_k,Gk)
			T = T + np.outer(t_k,Pk);# + np.outer(t_0,Pk); # Base States Added
			C = C +	np.outer(t_k,Gk);# + np.outer(t_0,Pk);
		else:
			PSI = PSI + np.outer(psi_k,Gk)
			T = T + np.outer(t_k,Pk)
			C = C +	np.outer(c_k,Pk)
		

	return PSI, T, C,T_00;

def Outer_Prod_V(R,theta,N_modes,X,d,Re_1,Re_2): # Fix this code to extend to the boundaries

	Psi_k, T_k, Omega_k = np.zeros(len(R)-2), np.zeros(len(R)-2), np.zeros(len(R)-2); # Obtained the required shape of numerical vector
	T_0, Omega_0 = np.zeros(len(R)-2), np.zeros(len(R)-2); 
	Gk, Pk = np.zeros(len(theta)), np.zeros(len(theta))
	mp.dps, eps = 20, 1e-4;

	s = (len(R)-2,len(theta));
	PSI, T, OMEGA = np.zeros(s), np.zeros(s), np.zeros(s); 

	## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
	G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
	P = lambda x,kk: legendre(kk,cos(x))
	
	A_T, B_T = -(1.0+d)/d, -1.0/d;
	#eta = Re_2/Re_1; 
	alpha = 1.0+d;
	at = -(Re_2*alpha - Re_1)/( (alpha**3.0) - 1.0); bt = -alpha*( Re_1*(alpha**2.0) - Re_2)/( (alpha**3.0) - 1.0); # Correct constants

	for ii in xrange(len(R)-2):
		T_0[ii] = -A_T/R[ii+1] + B_T;
		Omega_0[ii] = at*(R[ii+1]**2) + bt/R[ii+1];	

	for k in xrange(N_modes):
		#print k
		#for jj in xrange(len(theta)):
		#	Gk[jj] = G(theta[jj],k);
		#	Pk[jj] = P(theta[jj],k);
		Gk = Geg_k[:,k];
		Pk = Leg_k[:,k];

		ind = k*3*(len(R)-2)
		nr = len(R)-2;
		Psi_k = X[ind:ind+nr];
		T_k = X[ind+nr:ind+2*nr];
		Omega_k = X[ind+2*nr:ind+3*nr];

		if k == 0:

			OMEGA = OMEGA +	np.outer(Omega_k,Gk)
			PSI = PSI + np.outer(Psi_k,Gk)
			T = T + np.outer(T_k,Pk) + np.outer(T_0,Pk);
		elif k ==1:
			OMEGA = OMEGA +	np.outer(Omega_k,Gk) + np.outer(Omega_0,Gk)
			PSI = PSI + np.outer(Psi_k,Gk)
			T = T + np.outer(T_k,Pk);
		else:
			OMEGA = OMEGA +	np.outer(Omega_k,Gk)
			PSI = PSI + np.outer(Psi_k,Gk)
			T = T + np.outer(T_k,Pk)

	# Add thermal base state	

	return PSI, T, OMEGA;

def Cast(R,theta,OMEGA,PSI,SIGMAT):
	#---- Repackage AA into omega[i,j] -------------
	nr, nth = len(R), len(theta);
	s = (nr,nth)
	omega, psi, Thermal = np.zeros(s),np.zeros(s),np.zeros(s);
	row, col = 0,0;
	for i in range(len(R)):
		col = i;
		for j in range(len(theta)): # Very Bottom and top rows must remain Zero therefore 
			omega[i,j] = OMEGA[col,0]
			psi[i,j] = PSI[col,0]
			Thermal[i,j] = SIGMAT[col,0]
			col = col + nr;
	return omega,psi,Thermal;

def Array_to_Vector_GSPACE(R,theta,psi,Thermal,omega):
	
	#---- Repackage AA into omega[i,j] -------------
	nr, nth = len(R), len(theta);
	s = (nr*nth,1)
	OMEGA,PSI,SIGMAT = np.zeros(s),np.zeros(s),np.zeros(s);
	jj = 0;
	for j in range(len(theta)): # Very Bottom and top rows must remain Zero therefore 
		for i in range(len(R)):
			OMEGA[jj,0] = omega[i,j];
			PSI[jj,0] = psi[i,j]; 
			SIGMAT[jj,0] = Thermal[i,j]; 
			jj+=1;

	return PSI,SIGMAT,OMEGA;	

def Plot_Package_CHE(R,theta,omega,psi,thermal,sigma): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	alpha = 1.0+sigma;

	'''#---- Repackage AA into omega[i,j] -------------
	nr, nth = len(R), len(theta)
	s = (nr,nth)
	omega, psi, thermal = np.zeros(s), np.zeros(s),np.zeros(s)
	row, col = 0,0
	for i in range(len(R)):
		col = i;
		for j in range(len(theta)): # Very Bottom and top rows must remain Zero therefore 
			omega[i,j] = OMEGA[col,t];
			psi[i,j] = PSI[col,t];
			thermal[i,j] = THERMAL[col,t];
			col = col + nr; 
	'''
	#if plot_out == True:
	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	###fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	# --------------- Plot Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)
	try:
		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		p1 = ax[0].contourf(theta,R,omega,RES) 
		ax[0].contour(theta,R,omega,RES)
		ax[0].contourf(2.0*np.pi-theta,R,omega,RES) 
		ax[0].contour(2.0*np.pi-theta,R,omega,RES)
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	except ValueError:
		pass
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#print omega.ax(axis=1).max()
	#print omega.min(axis=1).min()
	ax[0].set_xlabel(r'$C_{max,min} = (%.3f,%.3f)$'%(omega.max(axis=1).max(),omega.min(axis=1).min()), fontsize=20) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$C$', fontsize=16, va='bottom')
	cbaxes = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- PSI Stream Function --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[1].contourf(theta,R,psi,RES) 
		ax[1].contour(theta,R,psi,RES)
		ax[1].contourf(2.0*np.pi-theta,R,psi,RES) 
		ax[1].contour(2.0*np.pi-theta,R,psi,RES) #-TT	
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[1].set_xlabel(r'$\Psi_{max,min} = (%.3f,%.3f)$'%(psi.max(axis=1).max(),psi.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$\Psi$', fontsize=16, va='bottom')
	#cbaxes1 = fig.add_axes([0.6, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- Temperature Field -----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		p3 = ax[2].contourf(theta,R,thermal,RES) #-TT
		ax[2].contour(theta,R,thermal,RES) #-TT	
		ax[2].contourf(2.0*np.pi-theta,R,thermal,RES) #-TT	
		ax[2].contour(2.0*np.pi-theta,R,thermal,RES) #-TT	
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	ax[2].set_xlabel(r'$T_{max,min} = (%.3f,%.3f)$'%(thermal.max(axis=1).max(),thermal.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T$', fontsize=16, va='bottom')
	cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar3 = plt.colorbar(p3, cax=cbaxes2)
	
	#branch = 'RL'; # 'SL' 'RL'
	#STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	'''
	if Pr >= 1.0:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr001.eps']) 
	'''	
	#plt.savefig(STR, format='eps', dpi=1800)
	plt.show()
				
	#return omega, ksi, psi	

def Plot_Package_Therm(R,theta,omega,psi,thermal,sigma,Re1,Ra,Pr): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	alpha = 1.0+sigma;

	#if plot_out == True:
	fig, ax = plt.subplots(1,2,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	#fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	# ---------------- PSI Stream Function --------------------       
	ax[0].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[0].contourf(theta,R,psi,RES) 
		ax[0].contour(theta,R,psi,RES)
		ax[0].contourf(2.0*np.pi-theta,R,psi,RES) 
		ax[0].contour(2.0*np.pi-theta,R,psi,RES) #-TT	
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[0].set_xlabel(r'$D^2 \psi_{max,min} = (%.3f,%.3f)$'%(psi.max(axis=1).max(),psi.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$D^2  \psi$', fontsize=16, va='bottom')
	cbaxes1 = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- Temperature Field -----------------
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		p3 = ax[1].contourf(theta,R,thermal,RES) #-TT
		ax[1].contour(theta,R,thermal,RES) #-TT	
		ax[1].contourf(2.0*np.pi-theta,R,thermal,RES) #-TT	
		ax[1].contour(2.0*np.pi-theta,R,thermal,RES) #-TT	
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	ax[1].set_xlabel(r'$T_{max,min} = (%.3f,%.3f)$'%(thermal.max(axis=1).max(),thermal.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$T$', fontsize=16, va='bottom')
	cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar3 = plt.colorbar(p3, cax=cbaxes2)
	
	'''
	branch = 'RL'; # 'SL' 'RL'

	if Pr >= 1.0:
		STR = "".join([branch,'_Thermal_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Thermal_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Thermal_Solution_Re',str(int(Re1)),'_Pr001.eps']) 	
	'''
	plt.tight_layout()
	plt.savefig("X_THERM.eps", format='eps', dpi=1200)
	plt.show()

#def MultiAnimate(R,theta,SIGMAT,OMEGA,KSI,PSI,alpha,sigma,dt,Ra,Pr,Re1=0.0,w_in=0.0,w_out=0.0,acc_rate1=0.0,acc_rate2=0.0):
def	VID(R,theta,N_Modes,X_VID,d,Ra,dt):
	sigma = d;
	alpha = 1.0+d;
	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20;
	azimuths = np.linspace(0,1e-11,NN)
	zeniths = np.linspace(0,3, NN )
	s = (NN,NN); ww = np.zeros(s)
	
	# Make space for title to clear
	#plt.subplots_adjust(top=0.8)
	plt.ion();
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif');

	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	#fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	R = R[1:-1]
	#psi, thermal, omega = Outer_Prod_V(R,theta,N_Modes,X_VID[:,100],d,Re_1,Re_2)
	PSI, T, C, T_0 = Outer_Prod_PTC(R,theta,N_Modes,X_VID[:,100],d)
	#PSI, T, C, T_0 = SPEC_TO_GRID(R,xx,theta,N_Modes,X,d)

	# --------------- Plot Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)
	try:
		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		p1 = ax[0].contourf(theta,R,C,RES) 
		ax[0].contour(theta,R,C,RES)
		ax[0].contourf(2.0*np.pi-theta,R,C,RES) 
		ax[0].contour(2.0*np.pi-theta,R,C,RES)
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	except ValueError:
		pass
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#print omega.ax(axis=1).max()
	#print omega.min(axis=1).min()
	ax[0].set_xlabel(r'$C_{max,min} = (%.3f,%.3f)$'%(C.max(axis=1).max(),C.min(axis=1).min()), fontsize=20) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$C$', fontsize=16, va='bottom')
	cbaxes = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- PSI Stream Function --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[1].contourf(theta,R,PSI,RES) 
		ax[1].contour(theta,R,PSI,RES)
		ax[1].contourf(2.0*np.pi-theta,R,PSI,RES) 
		ax[1].contour(2.0*np.pi-theta,R,PSI,RES) #-TT	
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[1].set_xlabel(r'$\Psi_{max,min} = (%.3f,%.3f)$'%(PSI.max(axis=1).max(),PSI.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$\Psi$', fontsize=16, va='bottom')
	#cbaxes1 = fig.add_axes([0.6, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- Temperature Field -----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		p3 = ax[2].contourf(theta,R,T,RES) #-TT
		ax[2].contour(theta,R,T,RES) #-TT	
		ax[2].contourf(2.0*np.pi-theta,R,T,RES) #-TT	
		ax[2].contour(2.0*np.pi-theta,R,T,RES) #-TT	
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	ax[2].set_xlabel(r'$T_{max,min} = (%.3f,%.3f)$'%(T.max(axis=1).max(),T.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T$', fontsize=16, va='bottom')
	cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar3 = plt.colorbar(p3, cax=cbaxes2)


	RR = R[1:-1];
	print "Vid len ",len(X_VID[0,:])
	for k in range(100):#len(X_VID[0,:])): # Evolve For the number of time steps evaluated
		fac = 60*k
		#psi, Thermal, OMEGA = Outer_Prod_V(R,theta,N_Modes,X_VID[:,10*k],d,Re_1,Re_2)
		psi, thermal, C, T_0 = Outer_Prod_PTC(R,theta,N_Modes,X_VID[:,fac],d)

		try:
			# Omega
			p1 = ax[0].contourf(theta,R,C,RES) 
			ax[0].contour(theta,R,C,RES)
			ax[0].contourf(2.0*np.pi-theta,R,C,RES) 
			ax[0].contour(2.0*np.pi-theta,R,C,RES)
			
			#cbaxes0 = fig.add_axes([0.1, 0.3, 0.01, 0.4]) # left, bottom, height, width
			#cbar0 = plt.colorbar(p0, cax=cbaxes0) #,fontsize=20)

			# PSI
			p2 = ax[1].contourf(theta,R,psi,RES) 
			ax[1].contour(theta,R,psi,RES)
			ax[1].contourf(2.0*np.pi-theta,R,psi,RES) 
			ax[1].contour(2.0*np.pi-theta,R,psi,RES) 
			
			#cbaxes1 = fig.add_axes([0.1, 0.3, 0.01, 0.4]) # left, bottom, height, width
			#cbar1 = plt.colorbar(p1, cax=cbaxes1) #,fontsize=20)
	
			# Temperature
			p3 = ax[2].contourf(theta,R,thermal,RES) #-TT
			ax[2].contour(theta,R,thermal,RES) #-TT	
			ax[2].contourf(2.0*np.pi-theta,R,thermal,RES) #-TT	
			ax[2].contour(2.0*np.pi-theta,R,thermal,RES) #-TT	

			#cbaxes2 = fig.add_axes([0.55, 0.3, 0.01, 0.4]) # left, bottom, height, width
			#cbar2 = plt.colorbar(p2, cax=cbaxes2) #, fontsize=20)
		except ValueError:  #raised if `y` is empty.
			pass

		# Inlcude Frame time	
		print 'Frame Number: ', fac
		Time = 10.0*float(fac)*dt     		
		st = r'$Time \quad (\alpha/r^2_1)t = \quad %4.1f $' % Time  #(0.25*float(k))#(k*dt)/w_ref
		plt.figtext(0.15, 0.1, st , color='black', fontsize = 20 ) #, weight='roman') #,size='x-small')

		
		plt.draw();
		plt.pause(0.01);
		#time.sleep(0.01)
		#fig.canvas.draw();
		#fig.canvas.flush_events();
		plt.clf()

def Spectral_Proj(R,theta,N_modes,OMEGA_NUM,PSI_NUM,T_NUM):

	# N_modes, Define Max Mode to cast on to

	#Define F_l(r) for each field, where psi_num = sum_{l=0}^N F_l(r)*(G_l)
	Omega_l, Psi_l, T_l = np.zeros((len(R),N_modes)), np.zeros((len(R),N_modes)), np.zeros((len(R),N_modes))

	# 1) Cast the vectors into 2D vectors 
	omega, psi, Thermal = Cast(R,theta,OMEGA_NUM,PSI_NUM,T_NUM) # All these vectors are Nr X Nth in dimensions
	omega_IP, psi_IP, Thermal_IP = omega, psi, Thermal

	for l in xrange(N_modes):
		print "Mode ",l,"\n"
		
		# a) Obtain the Projection Vectors for each mode
		Psi_IP, T_IP = Inner_Prod(R,theta,l);
		Omega_IP = Psi_IP; # Same as it goes like ~ sin \theta d/d \theta [ P_l(cos \theta) ]
		
		# b) Inner product these vectors up to N_theta mode
		for ii in xrange(len(R)):
			Omega_l[ii,l] = np.dot(omega_IP[ii,:],Omega_IP)
			Psi_l[ii,l] = np.dot(psi_IP[ii,:],Psi_IP)
			T_l[ii,l] = np.dot(Thermal_IP[ii,:],T_IP)

		Psi_OP, T_OP = Outer_Prod(R,theta,l);
		Omega_OP = Psi_OP;	

		# c) Update the casted vector	
		omega_IP = omega_IP - 	np.outer(Omega_l[:,l],Omega_OP[:])
		Psi_IP = Psi_IP - np.outer(Psi_l[:,l],Psi_OP[:])
		Thermal_IP = Thermal_IP - np.outer(T_l[:,l],T_OP[:])

		print "T_l ",T_l[:,l],"\n"

	return 	Psi_l, T_l, Omega_l

def Plot_Package(R,theta,SIGMAT,OMEGA,PSI,t): # Returns Array accepted by contourf - function
	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,0.001, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	#---- Repackage AA into omega[i,j] -------------
	nr = len(R); nth = len(theta); s = (nr,nth)
	psi, Thermal, omega = np.zeros(s),np.zeros(s),np.zeros(s)

	alpha = 1.0 + abs(R[-1] - R[0])

	L = OMEGA[:,t].shape
	row, col = 0,0;
	for i in range(len(R)):
		col = i
		for j in range(len(theta)): 
			omega[i,j] = OMEGA[col,t]
			#ksi[i,j] = KSI[col,t]*(1.0/(R[i]*math.sin(theta[j])) )
			psi[i,j] = PSI[col,t]
			Thermal[i,j] = SIGMAT[col,t]
			col = col + nr;
	

	#if plot_out == True:
	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))
	#fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(18,10))  
	#fig.suptitle(r'Reynolds Number $Re_{inner} = %s$, Gap width $\sigma = %s$'%(Re1,sigma), fontsize=16)      
	#print ax
	# --------------- Plot Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p1 = ax[0].contourf(theta,R,omega,RES) 
		ax[0].contour(theta,R,omega,RES)
		ax[0].contourf(2.0*np.pi-theta,R,omega,RES) 
		ax[0].contour(2.0*np.pi-theta,R,omega,RES)
	except ValueError:
		pass	
	#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	#ax[0].set_position([0.0,0.525, 0.34, 0.34])
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[0].set_xlabel(r'\textit{Polar Angle } ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$\Omega$$', fontsize=30, va='bottom')
	cbaxes = fig.add_axes([0.03, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- PSI vorticty --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[1].contourf(theta,R,psi,RES) 
		ax[1].contour(theta,R,psi,RES)
		ax[1].contourf(2.0*np.pi-theta,R,psi,RES) 
		ax[1].contour(2.0*np.pi-theta,R,psi,RES)	
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	
	#ax[1].set_position([0.3,0.525, 0.34, 0.34])
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[1].set_xlabel(r'\textit{Polar Angle } ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$\psi$', fontsize=30, va='bottom')
	cbaxes1 = fig.add_axes([0.638, 0.01, 0.0125, 0.35]) # left, bottom, height, width
	cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- T Temperature Field -----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		#p3 = ax[2].contourf(theta,R,psi,5,levels = np.arange(-0.0005, 0.0005, 0.0001)) 
		p3 = ax[2].contourf(theta,R,Thermal,RES) #-TT
		ax[2].contour(theta,R,Thermal,RES) #-TT	
		ax[2].contourf(2.0*np.pi-theta,R,Thermal,RES) #-TT	
		ax[2].contour(2.0*np.pi-theta,R,Thermal,RES)
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	#ax[2].set_position([0.64,0.525, 0.34, 0.34])
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[2].set_xlabel(r'\textit{Polar Angle } ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T$', fontsize=30, va='bottom')
	cbaxes3 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar3 = plt.colorbar(p3, cax=cbaxes3)
	
	'''if SAVE == True:
		st = "%6.1f"%Re1
		Some_String = ['Test_Re_',st,'.eps']
    		Stringy = ['/home/mannixp/Dropbox/Imperial/MResProject/Python_Codes/Results/AnalyticCompare 100.0']
    		os.chdir("".join(Stringy))	
		plt.savefig("".join(Some_String), format='eps', dpi=1000)'''
	plt.show()
	return omega, psi, Thermal

def SPECIAL_Plot_Package_CHE(R_e,theta_e,sigma_e,R_o,theta_o,sigma_o,psi_r,omega_r,psi_e,thermal_e,psi_o,thermal_o): #Re1,Ra,Pr): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	alpha = 1.0+sigma_e;
	R = R_e; theta = theta_e;

	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	###fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	# --------------- Plot Psi/Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)
	try:
		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		p1 = ax[0].contourf(theta,R,psi_r,RES) 
		ax[0].contour(theta,R,psi_r,RES)
		ax[0].contourf(2.0*np.pi-theta,R,omega_r,RES) 
		ax[0].contour(2.0*np.pi-theta,R,omega_r,RES)
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	except ValueError:
		pass
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#print omega.ax(axis=1).max()
	#print omega.min(axis=1).min()
	ax[0].set_xlabel(r'a) $\psi_{max,min} = (%.2f,%.2f)$'%(psi_r.max(axis=1).max(),psi_r.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$\Omega_0(r,\theta) \quad\quad \quad\quad \psi_0(r,\theta)$', fontsize=16, va='bottom')
	#cbaxes = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- Psi/Temp EVEN --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[1].contourf(theta,R,psi_e,RES); # Psi_Even
		ax[1].contour(theta,R,psi_e,RES); # Psi_Even
		ax[1].contourf(2.0*np.pi-theta,R,thermal_e,RES); #TT_Even	
		ax[1].contour(2.0*np.pi-theta,R,thermal_e,RES) # TT_Even	
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[1].set_xlabel(r'b) $\psi_{max,min} = (%.2f,%.2f)$'%(psi_e.max(axis=1).max(),psi_e.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$T(r,\theta) \quad\quad \quad\quad \psi(r,\theta)$', fontsize=16, va='bottom')
	#cbaxes1 = fig.add_axes([0.6, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar2 = plt.colorbar(p2, cax = cbaxes1)

	alpha = 1.0+sigma_o;
	R = R_o; theta = theta_o;

	# ----------------- Psi/Temp ODD-----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		p3 = ax[2].contourf(theta,R,psi_o,RES) # Psi_ODD
		ax[2].contour(theta,R,psi_o,RES) # Psi_ODD
		ax[2].contourf(2.0*np.pi-theta,R,thermal_o,RES) # TT_ODD	
		ax[2].contour(2.0*np.pi-theta,R,thermal_o,RES) # TT_ODD		
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	ax[2].set_xlabel(r'c) $\psi_{max,min} = (%.2f,%.2f)$'%(psi_o.max(axis=1).max(),psi_o.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T(r,\theta) \quad\quad \quad\quad \psi(r,\theta)$', fontsize=16, va='bottom')
	#cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar3 = plt.colorbar(p3, cax=cbaxes2)
	
	'''
	branch = 'RL'; # 'SL' 'RL'

	if Pr >= 1.0:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr001.eps']) 	
	'''
	STR = "".join(['3_Solution_Pr10.eps'])
	plt.savefig(STR, format='eps', dpi=1800)
	plt.show()
				
	#return omega, ksi, psi		


def Plot_Package_CHE_RAD(R,theta,omega,psi,thermal,sigma,Re1,Ra,Pr,ij,pos): # Returns Array accepted by contourf - function

	#fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	###fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	fig, axs = plt.subplots(3, sharex=True, sharey=True,figsize=(8,6))
	#plt.title(r'Radial Profile',fontsize=20)
	#axs[0].ylabel(r'$\frac{X}{||X||}$',fontsize=20)
	#axs[0].xlabel(r'$r$',fontsize=20)

	axs[0].plot(R,psi[:,ij],'k--',label = r'$\psi(r,\theta =%s )$'%pos)
	axs[1].plot(R,thermal[:,ij],'b--',label = r'$T(r,\theta =%s )$'%pos)
	axs[2].plot(R,omega[:,ij],'y--',label = r'$\Omega(r,\theta =%s )$'%pos)
	

	'''
	branch = 'RL'; # 'SL' 'RL'

	if Pr >= 1.0:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr001.eps']) 	
	plt.savefig(STR, format='eps', dpi=1800)
	'''
	
	#plt.xlim([R[0],R[-1]])
	plt.legend()
	plt.show()
				
	#return omega, ksi, psi	


def SPEC_TO_GRID_EIG(R,xx,theta,N_modes,X,d,Re_1,Re_2,Pr): # Fix this code to extend to the boundaries

	Nr = len(R)
	Psi_k, T_k, Omega_k = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R)); # Obtained the required shape of numerical vector
	T_0, Omega_0 = np.zeros(len(R)), np.zeros(len(R));
	#T_00 = np.zeros(len(R)); Omega_00
	
	Gk, Pk = np.zeros(len(theta)), np.zeros(len(theta))
	mp.dps, eps = 20, 1e-4;


	# Define New Radial Grid and Vectors
	s = (len(xx),len(theta));
	PSI, T, OMEGA = np.zeros(s), np.zeros(s), np.zeros(s); 

	## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
	G = lambda x,kk: -sin(x)*sin(x)*gegenbauer(kk-1,1.5,cos(x))
	P = lambda x,kk: legendre(kk,cos(x))
	
	A_T, B_T = -(1.0+d)/d, -1.0/d;
	alpha = 1.0+d; #eta = Re_2/Re_1; 
	at = -(Re_2*alpha - Re_1)/( (alpha**3.0) - 1.0); bt = -alpha*( Re_1*(alpha**2.0) - Re_2)/( (alpha**3.0) - 1.0); # Correct constants

	for ii in xrange(len(R)):
		T_0[ii] = -A_T/R[ii] + B_T;
		Omega_0[ii] = Pr*( at*(R[ii]**2) + bt/R[ii] );	 #Pr*

	# Interpolate	
	#Psi_0 = np.polyval(np.polyfit(R,Psi_0,Nr),xx)
	t_0 = np.polyval(np.polyfit(R,T_0,Nr),xx)
	omega_0 = np.polyval(np.polyfit(R,Omega_0,Nr),xx)

	for k in xrange(N_modes):
		#print " k ",k,"\n"
		#for jj in xrange(len(theta)):
		#	Gk[jj] = G(theta[jj],k);
		#	Pk[jj] = P(theta[jj],k);
		Gk = Geg_k[:,k];
		Pk = Leg_k[:,k];

		nr = len(R)-2; ind = k*3*nr;
		Psi_k[1:-1] = X[ind:ind+nr,0].reshape(nr);
		T_k[1:-1] = X[ind+nr:ind+2*nr,0].reshape(nr);
		Omega_k[1:-1] = X[ind+2*nr:ind+3*nr,0].reshape(nr);

		psi_k = np.polyval(np.polyfit(R,Psi_k,Nr),xx)
		t_k = np.polyval(np.polyfit(R,T_k,Nr),xx)
		omega_k = np.polyval(np.polyfit(R,Omega_k,Nr),xx)

		if k==0:
			T_00 = np.outer(t_0,Pk)
		if k == 0:

			OMEGA = OMEGA +	np.outer(omega_k,Gk)
			PSI = PSI + np.outer(psi_k,Gk)
			T = T + np.outer(t_k,Pk)+ np.outer(t_0,Pk); # Thermal Base State Added
		elif k ==1:
			OMEGA = OMEGA +	np.outer(omega_k,Gk)+ np.outer(omega_0,Gk); # Rotating Base State Added
			PSI = PSI + np.outer(psi_k,Gk)
			T = T + np.outer(t_k,Pk);
		else:
			OMEGA = OMEGA +	np.outer(omega_k,Gk)
			PSI = PSI + np.outer(psi_k,Gk)
			T = T + np.outer(t_k,Pk)
		
		#OMEGA = OMEGA +	np.outer(Omega_k,Gk)
		#PSI = PSI + np.outer(Psi_k,Gk)
		#T = T + np.outer(T_k,Pk)	
	# Add thermal base state	

	return PSI, T, OMEGA,T_00;



def Plot_Package_CHE_HOPF(R,theta,omega,psi,thermal,omega_imag,psi_imag,thermal_imag,sigma,Re1,Ra,Pr): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	alpha = 1.0+sigma;

	'''#---- Repackage AA into omega[i,j] -------------
	nr, nth = len(R), len(theta)
	s = (nr,nth)
	omega, psi, thermal = np.zeros(s), np.zeros(s),np.zeros(s)
	row, col = 0,0
	for i in range(len(R)):
		col = i;
		for j in range(len(theta)): # Very Bottom and top rows must remain Zero therefore 
			omega[i,j] = OMEGA[col,t];
			psi[i,j] = PSI[col,t];
			thermal[i,j] = THERMAL[col,t];
			col = col + nr; 
	'''
	#if plot_out == True:
	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	###fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	# --------------- Plot Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)
	try:
		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		p1 = ax[0].contourf(theta,R,omega,RES)
		ax[0].contourf(theta,R,omega,RES) 
		ax[0].contour(theta,R,omega,10, colors='k', linestyles='-')
		ax[0].contourf(2.0*np.pi-theta,R,omega_imag,RES) 
		ax[0].contour(2.0*np.pi-theta,R,omega_imag,10, colors='k', linestyles='-')
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	except ValueError:
		pass
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#print omega.ax(axis=1).max()
	#print omega.min(axis=1).min()
	#ax[0].set_xlabel(r'$\Omega_{max,min} = (%.3f,%.3f)$'%(omega.max(axis=1).max(),omega.min(axis=1).min()), fontsize=20) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$\Omega_{unstable} \quad\quad \Omega_{stable}$', fontsize=30, va='bottom')
	#cbaxes = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- PSI Stream Function --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:		
		p2 = ax[1].contourf(theta,R,psi,RES)
		ax[1].contourf(theta,R,psi,RES) 
		ax[1].contour(theta,R,psi,10, colors='k', linestyles='-')
		ax[1].contourf(2.0*np.pi-theta,R,psi_imag,RES) 
		ax[1].contour(2.0*np.pi-theta,R,psi_imag,10, colors='k', linestyles='-')
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[1].set_xlabel(r'$\Psi_{max,min} = (%.3f,%.3f)$'%(psi.max(axis=1).max(),psi.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$\Psi_{unstable} \quad\quad \Psi_{stable}$', fontsize=30, va='bottom')
	#cbaxes1 = fig.add_axes([0.6, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- Temperature Field -----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours		
		p3 = ax[2].contourf(theta,R,thermal,RES) #-TT
		ax[2].contourf(theta,R,thermal,RES)
		ax[2].contour(theta,R,thermal,5,colors='k', linestyles='-')	
		ax[2].contourf(2.0*np.pi-theta,R,thermal_imag,RES) #-TT	
		ax[2].contour(2.0*np.pi-theta,R,thermal_imag,5, colors='k', linestyles='-')
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	#ax[2].set_xlabel(r'$T_{max,min} = (%.3f,%.3f)$'%(thermal.max(axis=1).max(),thermal.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T_{unstable} \quad\quad T_{stable}$', fontsize=30, va='bottom')
	#cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar3 = plt.colorbar(p3, cax=cbaxes2)
	'''
	branch = 'RL'; # 'SL' 'RL'

	if Pr >= 1.0:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr001.eps']) 	
	'''
	plt.tight_layout()
	STR = "q_POINT_Re5.eps"
	plt.savefig(STR, format='eps', dpi=1800)
	plt.show()
				
	#return omega, ksi, psi		