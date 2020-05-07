#! /usr/bin/env python


# Method improved to increase integration degree, and use 'tanh-sinh' method adept to capturing endpoint singularities?

## Speed up by taking l**2 or m**2 terms to be all even and l*m to be odd therefore every second term

### NOTE: This must take (k) and (y) as from zero (n) goes from zero also

import os, sys
from mpmath import *
from sympy	import *
import numpy as np
import scipy.special as sp

from sympy.physics.wigner import gaunt
from sympy.physics.wigner import wigner_3j


mp.dps = 40; #mp.pretty = True
eps = 1e-14;

### ~~~~~~~~~~~~~~~~~~ A NOTE ON CONVENTIONS ADOPTED ~~~~~~~~~~~~~~~
def norm_geg(n):
	alpha = 1.5
	return ( ( float(sp.factorial(n))*(n + alpha)*(sp.gamma(alpha)**2) )/( np.pi*(2.0**(1.0-2.0*alpha))*sp.gamma(n +2.0*alpha) ) )

## ~~~~~~~~~~~~~~~~~ Gegenbauer & Legendre Polynomials ~~~~~~~~~~~~~~~~
G = lambda x,l: -sin(x)*sin(x)*gegenbauer(l-1,1.5,cos(x))
P = lambda n,x: legendre(n,x)
P_lm = lambda l,m,x: legenp(l, m, x); 

#sphere = [0.0,float(pi)], [0.0,float(2*pi)]
#ds = lambda x: sin(x)
#Y = lambda l,m,theta,phi: spherharm(l,m,theta,phi)
#Y_p = lambda l,m,theta,phi: m*cot(theta)*spherharm(l,m,theta,phi) + sqrt((l-m)*(l+m+1))*exp(-j*phi)*spherharm(l,m+1,theta,phi)

inter = [-0.999,-0.001,0.001,0.999];
#inter = [-1.0,1.0];
N = 100;
dx = 2.0/N;
xx = np.linspace(-0.99,0.99,N);

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

# w1 integrates int_{0}^{pi} P_l^m P_p^q P_n^s \sin \theta d \theta
def w1_coeff(l,p,n,m,q,s):
	beta = 0.0;
	if (l+p+n)%2 == 0:
		f = lambda x: P_lm(l,m,x)*P_lm(p,q,x)*P_lm(n,s,x)
		for ii in xrange(N):
			beta += float(f(xx[ii]))*dx;
		#beta = float(mpmath.quad(lambda x: P_lm(l,m,x)*P_lm(p,q,x)*P_lm(n,s,x),inter, method='tanh-sinh', maxdegree=20)) #
	else:
		beta = 0;	
	return 0.5*(2*n+1)*(sp.factorial(n-s)/sp.factorial(n+s))*beta;
			
def w1_coeffs(l,p,n,m,q,s):
	w1 = 0.0;
	if (l >= abs(m)) and (p >= abs(q)) and (n >= abs(s)):
		if (l+p+n)%2 == 0: 
			fac = np.sqrt( (sp.factorial(l+m)/sp.factorial(l-m))*(sp.factorial(p+q)/sp.factorial(p-q))*(sp.factorial(n-s)/sp.factorial(n+s)));
			w1 = ((-1)**s)*float(wigner_3j(l,p,n,0,0,0)*wigner_3j(l,p,n,m,q,-s))*fac;
	return (2*n+1)*w1.real;

# w2 integrates int_{0}^{pi} d(P_l^m) d(P_p^q) P_n^s \sin \theta d \theta
def w2_coeff(l,p,n,m,q,s):
	beta = 0.0
	if (l+p+n)%2 == 0:
		f = lambda x: 0.25*( (l+m)*(l-m+1)*(p+q)*(p-q+1)*P_lm(l,m-1,x)*P_lm(p,q-1,x) - (l+m)*(l-m+1)*P_lm(l,m-1,x)*P_lm(p,q+1,x) - (p+q)*(p-q+1)*P_lm(l,m+1,x)*P_lm(p,q-1,x) + P_lm(l,m+1,x)*P_lm(p,q+1,x) )*P_lm(n,s,x)
		for ii in xrange(N):
			beta += float(f(xx[ii]))*dx;
		#beta = float(integrate(f,(x,-1,1)))
		#beta = float(mpmath.quad(f,inter, method='tanh-sinh', maxdegree=40)) #
	else:
		beta = 0;	
	return 0.5*(2*n+1)*(sp.factorial(n-s)/sp.factorial(n+s))*beta; #

def w2_coeffs(l,p,n,m,q,s):
	a,b,c,d = 0,0,0,0; fac = 0.0; w2 = 0.0

	if (l >= abs(m)) and (p >= abs(q)) and (n >= abs(s)): 
		#if ( abs(l-p) <= n <= l+p):
		if (l+p+n)%2 == 0:
			
			fac = np.sqrt( (sp.factorial(l+m)/sp.factorial(l-m))*(sp.factorial(p+q)/sp.factorial(p-q))*(sp.factorial(n-s)/sp.factorial(n+s)) );
			
			
			a = ((-1)**s)*float(wigner_3j(l,p,n,0,0,0)*wigner_3j(l,p,n,m-1,q-1,-s))*np.sqrt( (l+m)*(l-m+1)*(p+q)*(p-q+1) )
			#print "Coeff a",a,"\n"
			d = ((-1)**s)*float(wigner_3j(l,p,n,0,0,0)*wigner_3j(l,p,n,m+1,q+1,-s))*np.sqrt( (l-m)*(l+m+1)*(p-q)*(p+q+1) )

			if (a!= 0) or (d !=0):
				print "non-zero"

			b = ((-1)**s)*float(wigner_3j(l,p,n,0,0,0)*wigner_3j(l,p,n,m-1,q+1,-s))*np.sqrt( (p-q)*(p+q+1)*(l+m)*(l-m+1) )
			c = ((-1)**s)*float(wigner_3j(l,p,n,0,0,0)*wigner_3j(l,p,n,m+1,q-1,-s))*np.sqrt( (p+q)*(p-q+1)*(l-m)*(l+m+1) )

			w2 = 2.0*(2*n+1)*((a-b-c+d)/4.0)*fac;

	return w2.real	


def w3_coeff(l,p,n,m,q,s):
	beta = 0.0
	#if (l+p+n)%2 == 0:
	f = lambda x: -P_lm(l,m,x)*P_lm(n,s,x)*( -(p+1)*x*P_lm(p,q,x) + (p-q+1)*P_lm(p+1,q,x)   )*(1/(x**2 - 1))
	for ii in xrange(N):
		beta += float(f(xx[ii]))*dx;
	#beta = float(integrate(f,(x,-1,1)))
	#beta = float(mpmath.quad(f,inter, method='tanh-sinh', maxdegree=40)) #
	#else:
	#	beta = 0;	
	return 0.5*(2*n+1)*(sp.factorial(n-s)/sp.factorial(n+s))*beta; #

def w3_coeffs(l,p,n,m,q,s):
	#a,b,c,d = 0,0,0,0; fac = 0.0; w2 = 0.0
	'''
	if (l >= abs(m)) and (p >= abs(q)) and (n >= abs(s)): 
		#if ( abs(l-p) <= n <= l+p):
		if (l+p+n)%2 == 0:
	'''		
	fac = np.sqrt( (sp.factorial(l+m)/sp.factorial(l-m))*(sp.factorial(p+q)/sp.factorial(p-q))*(sp.factorial(n-s)/sp.factorial(n+s)) );
	
	
	a = float(wigner_3j(l,p-1,n,0,0,0)*wigner_3j(l,p-1,n,m,-2,-s))*(p**4 - 2.0*p**3 - p**2 + 2*p);

	b = float(wigner_3j(l,p-1,n,0,0,0)*wigner_3j(l,p-1,n,m, 2,-s));
	
	c = float(wigner_3j(l,p-1,n,0,0,0)*wigner_3j(l,p-1,n,m, 0,-s))*2.0*(p**2+p);

	w2 = (2*n+1)*((-1)**s)*(-(a+b+c)/4.0)*fac;

	return w2.real	





def INNER(l,m): # Normalization od integral

	return 0.5*(2*l+1)*(sp.factorial(l-m)/sp.factorial(l+m));

# Toroidal terms

def alpha1(l,n,m):
	
	if n == l:		
		return m*(l**2 + l - 2); #*INNER(n,m);
	else:
		return 0.0;	

def alpha2_dx(l,n,m):

	beta = 0.0
	#beta = float(mpmath.quad( lambda x: P_lm(n,m,x)*( l*(l+1)*(  (l-m+1)*P_lm(l+1, m, x) - x*P_lm(l,m,x)*(l-1) ) ), [-1.0,1.0], method='tanh-sinh', maxdegree=100)	)
	f = lambda x: P_lm(n,m,x)*( l*(l+1)*(  (l-m+1)*P_lm(l+1, m, x) - x*P_lm(l,m,x)*(l-1) ) );
	for ii in xrange(N):
		beta += float(f(xx[ii]))*dx;
	return INNER(n,m)*beta;

def alpha2(l,n,m): # Correct

	if n == l+1:		
		return ( l*(l+1.0)*(l+2.0)*(l-m+1.0) )/(2.0*l+1.0)#INNER(n,m)*
	elif n == l-1:		
		return ( -l*(l+1.0)*(l-1.0)*(l+m) )/(2.0*l+1.0); #INNER(n,m)*
	else:
		return 0.0;	


def alpha3_dx(l,n,m):

	beta = 0.0
	f = lambda x: P_lm(n,m,x)*( -2*( (l-m+1)*P_lm(l+1, m, x) + x*P_lm(l, m, x)*(l-1)*(l+1)) )
	for ii in xrange(N):
		beta += float(f(xx[ii]))*dx;
	return round(INNER(n,m)*beta,5);

def alpha3(l,n,m):

	if n == l+1:		
		return ( -2/(2*l+1) )*l*(l+2)*(l-m+1); #*INNER(n,m);
	elif n == l-1:		
		return ( -2/(2*l+1) )*(l-1)*(l+1)*(l+m); #*INNER(n,m);	
	else:
		return 0.0;	


# Poloidal Psi' Terms

def beta2_dx(l,n,m):

	beta = 0.0
	f = lambda x: P_lm(n,m,x)*(-2.0)*( -(l**2+l+1)*(l-m+1)*P_lm(l+1, m, x)+x*P_lm(l, m, x)*(l+1)*( (l-1)**2 ) )
	for ii in xrange(N):
		beta += float(f(xx[ii]))*dx;
	return round(INNER(n,m)*beta,5);

def beta2(l,n,m):

	if n == l-1:
		return ( (l+1)*( (l-1)**2 )*(l+m) )*(-2/(2*l+1)); #*INNER(n,m)
	elif n == l+1:
		return ( -l*( (l+2)**2 )*(l-m+1) )*(-2/(2*l+1) ); #*INNER(n,m)
	else:
		return 0.0	

# Poloidal Phi' Terms

def gamma1_dx(l,n,m):
	beta = 0.0
	f = lambda x: m*l*(l+1)*P_lm(n,m,x)*P_lm(l, m, x);
	for ii in xrange(N):
		beta += float(f(xx[ii]))*dx;
	return INNER(n,m)*beta;

def gamma1(l,n,m):		
	if n == l:		
		return m*l*(l+1); #*INNER(n,m)
	else:
		return 0.0;	

def gamma2(l,n,m):	
	if n == l:		
		return (l**2 + l -2)*m*l*(l+1); #*INNER(n,m)
	else:
		return 0.0;	

def gamma3(l,n,m):	
	if n == l:		
		return m; #*INNER(n,m)
	else:
		return 0.0;		

def gamma4(l,n,m):	
	if n == l:		
		return -(l**2 + l -2)*m; #*INNER(n,m)
	else:
		return 0.0;					



'''

# Fix m
m = 5;

# Vary l,n
N = 20;
for p in xrange(N):
	print " p ",p
	print "# ~~~~~~~~~ # ~~~~~~~~~~~ # ~~~~~~~~~~~ #"
	for l in xrange(N):
		l = float(l+m);
		print " l ",l
		print "# ~~~~~~~~~ # ~~~~~~~~~~~ # ~~~~~~~~~~~ #"
		for n in xrange(N):
			n = float(n+m);
			print "n ",n
			#print "Riemman dx ",w3_coeff(l,p,n,m,0,m)
			print "Exact ",w3_coeffs(l,p,n,m,0,m)

	print "# ~~~~~~~~~ # ~~~~~~~~~~~ # ~~~~~~~~~~~ # \n"	
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~

'''
## Calculate and save the coefficients

N_modes = 30+1;
s = (N_modes,N_modes,N_modes);
aPsi = np.zeros(s); bPsi = np.zeros(s);
aPsi_Om = np.zeros(s); bPsi_Om = np.zeros(s);
aTt = np.zeros(s); bTt = np.zeros(s);
aOm = np.zeros(s); bOm = np.zeros(s);

for n in xrange(N_modes):

	print "Mode l = ",n,"\n"
	for l in xrange(N_modes):

		for m in xrange(N_modes):

				if (l+m+n)%2 == 0:
	
					if (abs(l-m) <= n) and (n <= abs(l+m)): 
						# Determine PSI coefficients
						aPsi[l,m,n] = aPSI(l,m,n); bPsi[l,m,n] = bPSI(l,m,n);
						aPsi_Om[l,m,n] = aPSI_OM(l,m,n); bPsi_Om[l,m,n] = bPSI_OM(l,m,n);

						# Determine Temperature Coefficients
						aTt[l,m,n] = aT(l,m,n); bTt[l,m,n] = bT(l,m,n);

						# Determine Omega Coefficients
						aOm[l,m,n] = aOM(l,m,n); bOm[l,m,n] = bOM(l,m,n);

np.isnan(aPsi);
np.isnan(bPsi); 
np.isnan(aPsi_Om);
np.isnan(bPsi_Om);

np.isnan(aTt);
np.isnan(bTt); 
np.isnan(aOm);
np.isnan(bOm);

Stringy = ['/home/mannixp/Dropbox/Imperial/PhD_Thermal/Rotating/Integrated_Modes/']
os.chdir("".join(Stringy))

np.save("aPsi.npy",aPsi); np.save("bPsi.npy",bPsi);
np.save("aPsi_Om.npy",aPsi_Om); np.save("bPsi_Om.npy",bPsi_Om);
np.save("aT.npy",aTt); np.save("bT.npy",bTt);
np.save("aOm.npy",aOm); np.save("bOm.npy",bOm);
'''

###~~~~~~~~~~~ Checking Methods ~~~~~~~~~~~~~~~~~~~~~~~~~##

'''k, l = 3,2
LEN = 7
AA = np.zeros(LEN) # Up to k+1 as we need P_0(x) -> P_k(x)
for n in xrange(LEN):
	AA[n] = gamma2_kl(k,l,n)
print "B1^n ", AA[:]'''

#Glmm = lambda x, mm: -sin(x)*sin(x)*gegenbauer(mm-1,1.5,cos(x))
#Plmm = lambda x, mm: legendre(mm,cos(x))


#~~~~~~~~ Legendre Series ~~~~~~~~~~~~~~~
#f_theta = lambda x: AA[0]*Plmm(x,0) + AA[1]*Plmm(x,1) + AA[2]*Plmm(x,2) + AA[3]*Plmm(x,3) + AA[4]*Plmm(x,4) + AA[5]*Plmm(x,5) + AA[6]*Plmm(x,6) #+ AA[7]*Plmm(x,7) #+ AA[8]*Plmm(x,8) + AA[10]*Plmm(x,10)  + AA[12]*Plmm(x,12) + AA[1]*Plmm(x,1) + AA[3]*Plmm(x,3) + AA[5]*Plmm(x,5) + AA[7]*Plmm(x,7) + AA[9]*Plmm(x,9) + AA[11]*Plmm(x,11) 

#~~~~~~~~ Gegenbauer Series ~~~~~~~~~~~~~~~
#f_theta = lambda x: AA[0]*Glmm(x,0) + AA[1]*Glmm(x,1) + AA[2]*Glmm(x,2) + AA[3]*Glmm(x,3) + AA[4]*Glmm(x,4) + AA[5]*Glmm(x,5) + AA[6]*Glmm(x,6) #+ AA[7]*Glmm(x,7) + AA[8]*Glmm(x,8) #+ AA[10]*Glmm(x,10)  + AA[12]*Glmm(x,12)   + AA[7]*Glmm(x,7) + AA[9]*Glmm(x,9) + AA[11]*Glmm(x,11) 

# ~~~~~~~~~ Test functions
#f1 = lambda x: k*(k+1.0)*Plmm(x,k)*Glmm(x,l)
#f2 = lambda x: -( l*(l+1.0)*Plmm(x,l) + (2.0*cos(x)*Glmm(x,l))/(sin(x)*sin(x)) )*Glmm(x,k)
#f3 = lambda x: (cos(x)/(sin(x)**2))*Glmm(x,l)*Glmm(x,k)

#f4 = lambda x: l*(l+1.0)*Plmm(x,l)*Plmm(x,k)
#f1 = lambda x: Plmm(x,1)
#f2 = lambda x: Plmm(x,1)
#mpmath.plot([f1,f2],[0.01,float(pi)-0.01])
