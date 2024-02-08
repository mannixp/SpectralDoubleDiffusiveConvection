import numpy as np
import matplotlib.pyplot as plt
import math
import h5py, sys


def SPEC_TO_GRID(R,xx,N_fm,X,d): # Fix this code to extend to the boundaries

	
	# 2) Normalization Weights, must be inverses
	x = np.linspace(0,np.pi,N_fm);
	
		
	Nr = len(R); t_0 = np.zeros(len(R)); 
	# Base State
	R_1 = 1./d; 
	R_2 = (1. + d)/d;

	A_T = (R_1*R_2)/(R_1 - R_2);
	B_T =      R_1 /(R_1 - R_2);

	for ii in range(Nr):
		t_0[ii] = (-A_T/R[ii] + B_T)
	TR_0 = np.polyval(np.polyfit(R,t_0,Nr),xx)
	T_0 = np.outer(TR_0,np.ones(N_fm));
	

	# A) Computational grid
	nr = len(R[1:-1])
	s = (nr,N_fm); N = nr*N_fm
	Vort = np.zeros(s); Temp = np.zeros(s); Conc = np.zeros(s)
	
	# 1) Transform vector to sq grid
	for ii in range(N_fm):	
		ind = ii*nr
		Vort[:,ii] = X[ind:ind+nr,0].reshape(nr);
			
		ind = N + ii*nr
		Temp[:,ii] = X[ind:ind+nr,0].reshape(nr);

		ind = 2*N + ii*nr
		Conc[:,ii] = X[ind:ind+nr,0].reshape(nr);

	
	#Temp[:,0] += t_0
	#Conc[:,0] += t_0
	# 2) Take the idct, idst of each radial level set
	Vort = idst(Vort/N_fm,type=2,axis=-1)
	Temp = idct(Temp/N_fm,type=2,axis=-1) # Assuming these must be normalized?
	Conc = idct(Conc/N_fm,type=2,axis=-1)


	# B) Visualisation grid
	s = (len(xx),N_fm);
	PSI, T, C = np.zeros(s), np.zeros(s), np.zeros(s); 	
	
	# 3) Polyfti
	for ii in range(N_fm):
		
		psi = np.hstack((0,Vort[:,ii],0));
		PSI[:,ii] = np.polyval(np.polyfit(R,psi,Nr),xx)
		
		t = np.hstack((0,Temp[:,ii],0))
		T[:,ii] = np.polyval(np.polyfit(R,t,Nr),xx)
		
		c = np.hstack((0,Conc[:,ii],0))
		C[:,ii] = np.polyval(np.polyfit(R,c,Nr),xx)
		
	return PSI, T, 0.*C,T_0;

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

def Plot_Package_CHE_sd(R,theta,omega,psi,thermal,sigma): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	d = sigma;

	GIRD = [0.5*(1./d),1./d,(1.+d)/d]

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
	ax[0].set_ylim(0,(1.+d)/d)
	ax[0].set_rgrids(GIRD, angle=345.,fontsize=12)
	#plt.rc('text', usetex=True)
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
	ax[1].set_ylim(0,(1.+d)/d)
	ax[1].set_rgrids(GIRD, angle=345.,fontsize=12)
	#plt.rc('text', usetex=True)
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
	ax[2].set_ylim(0,(1.+d)/d)
	ax[2].set_rgrids(GIRD, angle=345.,fontsize=12)
	#plt.rc('text', usetex=True)
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

def SPECT_INT(D,f):

	Dt = D[0:-1,0:-1];
	return np.linalg.solve(Dt,f[0:-1])[0];#?

def Plot_Package_SPLIT(R,theta,psi,C,d): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	GIRD = [0.5*(1./d),1./d,(1.+d)/d]
	
	fig1, ax1 = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10,8))
	
	ax1.contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax1.contourf(theta, R, psi,25,cmap="RdBu_r") 
		pp = ax1.contourf(2.*np.pi - theta,R, C,25,cmap="RdBu_r")
		ax1.contour(theta,R, psi,8,cmap="RdBu_r")#colors = 'k',linewidths=0.7)
		ax1.contour(2.*np.pi - theta, R, C,8,cmap="RdBu_r")#, colors = 'k',linewidths=0.7)
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)		
	except ValueError:
		pass
	ax1.set_theta_zero_location("S")
	ax1.bar(math.pi, 0.0 )

	ax1.set_ylim(0,(1.+d)/d)
	ax1.set_rgrids(GIRD, angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax1.set_xlabel(r'\textit{Polar Angle } ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax1.set_title(r'$T(r,\theta) + T_0(r) \quad\quad \quad\quad  \Psi(r,\theta)$ ',fontsize=20, va='bottom')
	#ax1.set_title(r'$C$', fontsize=20, va='bottom')
	plt.tight_layout()
	plt.savefig('SPLIT_PSI_C.eps', format='eps', dpi=1800)
	plt.show()	


	#plt.savefig(STR, format='eps', dpi=1800)
	plt.show()
				
	#return omega, ksi, psi			


# All checked below here
	
RES = 25; 

def Energy(filename,frame=-1):

	"""

	Plot the energy in each longitudinal mode by integrating 

	f_k = 1/d int_r f_k(r) dr	

	out the radial dependency.

	Inputs:

	X    - vector of (psi,T,S)^n at one time-instant
	N_fm - int number of latitudinal modes
	R    - vector of radial collocation points


	Returns:

	None
	"""

	f = h5py.File(filename, 'r');
	
	X    = f['Checkpoints/X_DATA'][frame,:];
	N_fm = f['Parameters']["N_fm"][()];
	N_r  = f['Parameters']["N_r"][()];
	f.close()
	
	from Matrix_Operators import X_to_Vecs
	ψ,T,S = X_to_Vecs(X,N_fm,N_r - 1)

	E_ψ = [ np.linalg.norm(ψ[:,k],2) for k in range(N_fm)]
	E_T = [ np.linalg.norm(T[:,k],2) for k in range(N_fm)]
	E_S = [ np.linalg.norm(S[:,k],2) for k in range(N_fm)]
	k   =   np.arange(0,N_fm,1) # sinusoids modes

	plt.figure(figsize=(8, 6))

	plt.semilogy(k,E_ψ, 'k.',label=r'$\hat{\psi} \sim \sin(k \theta)$')
	plt.semilogy(k,E_T, 'r.',label=r'$\hat{T}$')
	plt.semilogy(k,E_S, 'b.',label=r'$\hat{S}$')

	plt.xlabel(r'Fourier-mode $k$', fontsize=26);
	plt.ylabel(r'$||X_k||$', fontsize=26)
	plt.xlim([0,N_fm])
	plt.grid()
	plt.legend()
	plt.tight_layout()
	plt.savefig("Energy_Spectra.pdf",format='pdf', dpi=200)
	plt.show()

	return None;

def Plot_Time_Step(filename,logscale=True):


	""""

	Plot out all scalar quantities vs. time

	Inputs:
	filename of h5py file

	Returns:

	None

	"""

	f = h5py.File(filename, 'r');

	st_pt = 0
	Time  = f['Scalar_Data/Time'][()][st_pt:-1]
	KE    = f['Scalar_Data/KE'][()][st_pt:-1]
	NuT   = f['Scalar_Data/Nu_T'][()][st_pt:-1]
	NuS   = f['Scalar_Data/Nu_S'][()][st_pt:-1]
	f.close()

	fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,figsize=(12, 6))

	# 1) Plot time-evolution
	if logscale == True:
		ax0.semilogy(Time,KE,'k-')

		#slope, intercept = np.polyfit(Time,np.log(KE),1)
		#print("Slope KE m=",slope,"\n");

		#slope, intercept = np.polyfit(Time,np.log(NuT),1)
		#print("Slope ||T||_2 m=",slope,"\n");

		##ax0.semilogy(Time,KE,'bo',markersize=0.3)
	else:
		ax0.plot(Time,KE,'k-')
	ax0.set_xlabel(r'Time $T$',fontsize=25)
	ax0.set_title(r'$\mathcal{E}$',fontsize=25)

	#ax0.set_xlim([Time[0],Time[-1]])

	if logscale == True:
		ax1.semilogy(Time,NuT,'k-')
	else:
		ax1.plot(Time,NuT,'k-')
	ax1.set_xlabel(r'Time $T$',fontsize=25)
	ax1.set_title(r'$Nu_T-1$',fontsize=25)

	if logscale == True:
		ax2.semilogy(Time,NuS,'k-')
	else:
		ax2.plot(Time,NuS,'k-')
	ax2.set_xlabel(r'Time $T$',fontsize=25)
	ax2.set_title(r'$Nu_S-1$',fontsize=25)
	
	
	plt.tight_layout()
	plt.savefig("Time_Series.pdf",format='pdf', dpi=1200)
	plt.show()

	return None;

def Spectral_To_Gridpoints(X, R,xx,N_fm,d): 

	from Main import Base_State_Coeffs
	from Transforms import IDCT, IDST
	from scipy.interpolate import interp1d
	from Matrix_Operators import X_to_Vecs


	A_T, B_T = Base_State_Coeffs(d)
	t_0      = np.asarray([ (-A_T/r + B_T) for r in R ]);
	TR_0 = np.interp(xx,R,t_0)
	T_0  = np.outer(TR_0,np.ones(N_fm));
	
	# 1) Transform vector to sq grid
	nr = len(R[1:-1]) 
	ψ_hat,T_hat,S_hat = X_to_Vecs(X,N_fm,nr)

	# 2) Take the idct, idst of each radial level set
	ψ = np.zeros((len(R),N_fm)); ψ[1:-1,:] = IDST(ψ_hat)
	T = np.zeros((len(R),N_fm)); T[1:-1,:] = IDCT(T_hat)
	S = np.zeros((len(R),N_fm)); S[1:-1,:] = IDCT(S_hat)

	# B) Visualisation grid
	fψ = interp1d(R, ψ, axis=0)
	fT = interp1d(R, T, axis=0)
	fS = interp1d(R, S, axis=0)
	
	return fψ(xx), fT(xx), fS(xx), T_0;

def Cartesian_Plot(filename,frame,Include_Base_State=True):


	if filename.endswith('.h5'):
		f    = h5py.File(filename, 'r');
		
		X    = f['Checkpoints/X_DATA'][frame,:];
		N_fm = f['Parameters']["N_fm"][()];
		N_r  = f['Parameters']["N_r"][()];
		d    = f['Parameters']["d"][()];

		f.close()

	if filename.endswith('.npy'):

		X = np.load(filename);
		N_fm = 50
		N_r = 20
		d = 0.353;
	

	from Matrix_Operators import cheb_radial

	R = cheb_radial(N_r,d)[1] 
	Theta_grid = np.linspace(0,np.pi,N_fm);  
	r_grid     = np.linspace(R[-1],R[0],50);

	PSI, T, S, T_0 = Spectral_To_Gridpoints(X, R,r_grid,N_fm,d)

	if Include_Base_State == True:
		T +=T_0; 
		S +=T_0; 

	# 1) Fix \theta labels to be [0,pi]
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,figsize=(12,8),dpi=1200)

	C_cntr  = ax1.contour( Theta_grid,r_grid,T,  RES, colors = 'k', linewidths=0.5,);
	C_cntrf = ax1.contourf(Theta_grid,r_grid,T,  RES, cmap="RdBu_r")
	fig.colorbar(C_cntrf, ax=ax1)#
	ax1.set_title(r'$T$',fontsize=20)

	P_cntr  = ax2.contour( Theta_grid,r_grid,PSI,RES, colors = 'k', linewidths=0.5);
	P_cntrf = ax2.contourf(Theta_grid,r_grid,PSI,RES, cmap="RdBu_r")
	fig.colorbar(P_cntrf, ax=ax2)#
	ax2.set_title(r'$\psi$',fontsize=20)


	T_cntr  = ax3.contour( Theta_grid,r_grid,S,  RES, colors = 'k',linewidths=0.5);
	T_cntrf = ax3.contourf(Theta_grid,r_grid,S,  RES, cmap="RdBu_r")
	fig.colorbar(T_cntrf, ax=ax3)#
	ax3.set_ylabel(r'$r$',fontsize=18)

	ax3.set_xlabel(r'$\theta$',fontsize=18)
	ax3.set_title(r'$S$',fontsize=18)

	plt.subplots_adjust(hspace=0.25)
	plt.tight_layout()
	plt.savefig("X_Frame.pdf",format='pdf', dpi=200)
	plt.show()

	return None;

def Uradial_plot(filename,frame):


	from Matrix_Operators import J_theta_RT
	from Matrix_Operators import cheb_radial
	from Transforms import IDCT, grid

	f    = h5py.File(filename, 'r');
	
	X    = f['Checkpoints/X_DATA'][frame,:];
	N_fm = f['Parameters']["N_fm"][()];
	N_r  = f['Parameters']["N_r"][()];
	d    = f['Parameters']["d"][()];

	f.close()

	D,R = cheb_radial(N_r,d); 
	nr = N_r -1;
	N  = N_fm*nr

	IR  	= np.diag(1./R[1:-1]);
	IR2 	= IR@IR;
	Jψ_hat  = J_theta_RT(X[0:N], nr,N_fm)
	u_r_hat = np.zeros((nr, N_fm)); 

	for ii in range(N_fm):
		ind_p = ii*nr; 
		u_r_hat[:,ii]  = IR2.dot(Jψ_hat[ind_p:ind_p+nr]);

	θ   = grid(N_fm);
	u_r = IDCT( u_r_hat);
	
	U = 0.*θ;
	for k in range(N_fm):
		f = np.zeros(len(R));
		f[1:-1] = u_r[:,k]
		U[k]    = np.linalg.solve(D[0:-1,0:-1],f[0:-1])[0];

	plt.figure(figsize=(8, 6))
	plt.plot(θ,U,'k-');
	plt.xlim([0,np.pi]);
	plt.xlabel(r"$\theta$",fontsize=22);
	plt.ylabel(r"$\bar{u}_r(\theta)$",fontsize=22);		
	plt.show();

	return None;

# Execute main
if __name__ == "__main__":

	# %% 
	print("Initialising the code for plotting ...")
	#%matplotlib inline
	
	# %%
	filename ='NewtonSolve_0.h5'; frame = -1;
	#Plot_Time_Step(filename,logscale=True);
	
	## %%
	Uradial_plot(filename,frame)
	Energy(filename, frame)
	Cartesian_Plot(filename, frame, Include_Base_State=False);
# %%
