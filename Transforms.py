
import numpy as np
from scipy.fftpack import dst,idst,dct,idct
import copy

N = 10;
x = np.asarray([np.pi*(.5 + i)/(2.*N) for i in range(N)])
f_cos = lambda k : np.cos(k*x)
f_sin = lambda k : np.sin(k*x)


k = 2
f = f_cos(k)
g = f_sin(k)

f_hat_in    = np.zeros(N)
f_hat_in[k] = 1

g_hat_in    = np.zeros(N)
g_hat_in[k] = 1

# --------------------
# Cosine transforms
# --------------------
print('~~~~ Cosine ~~~~~~')

# -- backwards scaling
# Scale sinusoid amplitudes to IDCT input
f_hat_scaled = copy.deepcopy(f_hat_in)
f_hat_scaled[1:] *=.5; 
# --

f     = idct(f_hat_scaled,type=2) 
f_hat = dct(f,type=2)

# -- forward scaling
# Scale DCT output to sinusoid amplitudes
f_hat    *= 1/N
f_hat[0] *= 0.5
# --

print('f_hat = ',f_hat_in)
print('f     = ',f)
print('f_hat = ',f_hat)

# %%
print('~~~~ Sine ~~~~~~')

# -- backwards scaling
# Scale sinusoid amplitudes to IDST input
g_hat_scaled = copy.deepcopy(g_hat_in)
g_hat_scaled *=.5; 
start = g_hat_scaled[0:N-1];
shift = g_hat_scaled[1:N  ];
np.copyto(start,shift);
g_hat_scaled[-1] = 0; # Drop Nyquist mode
# --

g     =idst(g_hat_scaled,type=2) 

g_hat = dst(g,type=2)

# -- forward scaling
# Scale DST output to sinusoid amplitudes
start = g_hat[0:N-1];
shift = g_hat[1:N  ];
np.copyto(shift,start);
g_hat[0] = 0
g_hat   *= 1/N
# --

print('g_hat = ',g_hat_in)
print('g     = ',g)
print('g_hat = ',g_hat)


# %%


# f_hat = dct(f,type=2)

# # -- forward scaling
# # Scale DCT output to sinusoid amplitudes
# f_hat    *= 1/N
# f_hat[0] *= 0.5
# # --

# # -- backwards scaling
# # Scale sinusoid amplitudes to IDCT input
# f_hat_scaled = copy.deepcopy(f_hat)
# f_hat_scaled[1:] *=.5; 
# # --

# f =idct(f_hat_scaled,type=2) 

# print('~~~~ Cosine ~~~~~~')
# print('f     = ',f_cos(k))
# print('f_hat = ',f_hat)
# print('f     = ',f)


# g_hat = dst(g,type=2)

# # -- forward scaling
# # Scale DST output to sinusoid amplitudes
# start = g_hat[0:N-1];
# shift = g_hat[1:N  ];
# np.copyto(shift,start);
# g_hat[0] = 0
# g_hat   *= 1/N
# # --

# # -- backwards scaling
# # Scale sinusoid amplitudes to IDST input
# g_hat_scaled = copy.deepcopy(g_hat)
# g_hat_scaled *=.5; 
# start = g_hat_scaled[0:N-1];
# shift = g_hat_scaled[1:N  ];
# np.copyto(start,shift);
# g_hat_scaled[-1] = 0; # Drop Nyquist mode
# # --

# g =idst(g_hat_scaled,type=2) 

# print('~~~~ Sine ~~~~~~')
# print('g     = ',f_sin(k))
# print('g_hat = ',g_hat)
# print('g     = ',g)