
import numpy as np
import copy
from   scipy.fftpack import dct,dst

L  = np.pi/2.;

def grid(N):

    dx = L/N;
    x  = [dx*(2.*i + 1.)/2 for i in range(N)]

    return np.asarray(x);

# Scalings
def sinusoid_to_IDCT(f_hat):

    """
    Scale sinusoid amplitudes to IDCT input    
    """

    f_hat_scaled      = copy.deepcopy(f_hat)
    f_hat_scaled[1:] *=.5;

    return f_hat_scaled;

def DCT_to_sinusoid(f_hat_scaled):

    """
    Scale DCT output to sinusoid amplitudes
    """

    f_hat     = copy.deepcopy(f_hat_scaled)
    N         = f_hat.shape[0] 
    f_hat    *= 1/N
    f_hat[0] *= 0.5

    return f_hat

def sinusoid_to_IDST(g_hat):
    """    
    Scale sinusoid amplitudes to IDST input
    """
    g_hat_scaled = copy.deepcopy(g_hat)
    g_hat_scaled *=0.5; 
    
    # N = g_hat.shape[0]
    # start = g_hat_scaled[0:N-1];
    # shift = g_hat_scaled[1:N  ];
    # np.copyto(start,shift);
    # g_hat_scaled[-1] = 0.; # Drop Nyquist mode
    
    return g_hat_scaled;

def DST_to_sinusoid(g_hat_scaled):

    """
    Scale DST output to sinusoid amplitudes
    """
    g_hat = copy.deepcopy(g_hat_scaled)
    N     = g_hat.shape[0]

    # start = g_hat[0:N-1];
    # shift = g_hat[1:N  ];
    # np.copyto(shift,start);
    # g_hat[0] = 0
    g_hat   *= 1/N

    return g_hat

# Transforms 
def DST(g,n=None,axis=-1):

    """
    Compute the DST of g and scale DST output
    to sinusoid amplitudes g_hat
    """
    g_hat_scaled = dst(g,type=2,axis=axis)
    if n == None:
        g_hat = DST_to_sinusoid(g_hat_scaled)
    else:
        g_hat = DST_to_sinusoid(g_hat_scaled)[0:n]

    return g_hat;

def IDST(g_hat,n=None,axis=-1):

    """
    Scale the sinusoid amplitudes f_hat to the IDST input
    and compute the DST
    """
    
    g_hat_scaled = sinusoid_to_IDST(g_hat)
    if n == None:
        g = dst(g_hat_scaled,type=3,axis=axis)
    else:
        g = dst(g_hat_scaled,type=3,axis=axis,n = n)

    return g;

def DCT(f,n=None,axis=-1):

    """
    Compute the DCT of f and scale DCT output
    to sinusoid amplitudes f_hat
    """
    f_hat_scaled = dct(f,type=2,axis=axis)
    if n == None:
        f_hat = DCT_to_sinusoid(f_hat_scaled)
    else:
        f_hat = DCT_to_sinusoid(f_hat_scaled)[0:n]

    return f_hat;

def IDCT(f_hat,n=None,axis=-1):

    """
    Scale the sinusoid amplitudes f_hat to the IDCT input
    and compute the DCT
    """
    
    f_hat_scaled = sinusoid_to_IDCT(f_hat);
    if n == None:
        f = dct(f_hat_scaled,type=3,axis=axis)
    else:
        f = dct(f_hat_scaled,type=3,axis=axis,n=n) 

    return f;

# Tests
def Test_Cosine_Transform(k,N):

    f_hat_in    = np.zeros(N)
    f_hat_in[k] = 1

    x    = grid(N)
    f_in = np.cos(((k*np.pi)/L)*x)

    print('~~~~ Cosine coefficient space to grid space~~~~~~')

    f     = IDCT(f_hat_in)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(x,f_in,'r:')
    plt.plot(x,f,'bo')
    plt.show()

    f_hat = DCT(f)
    
    for a,b in zip(np.round(f_hat,12),f_hat_in):
        assert a == b

    print('~~~~ Cosine grid space to coefficient space~~~~~~')

    f_hat = DCT(f_in)
    f     = IDCT(f_hat) 
    
    for a,b in zip(np.round(f,12),np.round(f_in,12)):
        assert a == b

    return None;

def Test_Sine_Transform(k,N):

    g_hat_in    = np.zeros(N)
    g_hat_in[k] = 1

    x     = grid(N)
    g_in  = np.sin((k+1)*x)

    # %%
    print('~~~~ Sine coefficient space to grid space~~~~~~')

    g    = IDST(g_hat_in)
    g_hat=  DST(g)

    for a,b in zip(np.round(g_hat,12),g_hat_in):
        assert a == b
    # print('g_hat = ',g_hat_in)
    # print('g     = ',g       )
    # print('g_hat = ',g_hat   )

    print('~~~~ Sine grid space to coefficient space~~~~~~')

    g_hat=  DST(g_in)
    g    = IDST(g_hat)

    for a,b in zip(np.round(g,12),np.round(g_in,12)):
        assert a == b

    # print('g     = ',g_in )
    # print('g_hat = ',g_hat)
    # print('g     = ',g    )

    return None;

# Test dealiasing
def Test_Cosine_Transform_deal(k,N):

    f_hat_in    = np.zeros(N)
    f_hat_in[k] = 2.1

    x    = grid(N)
    f_in = 2.1*np.cos(k*x)

    print('~~~~ Cosine coefficient space to grid space~~~~~~')

    f     = IDCT(f_hat_in,n=(3*N)//2) 
    f_hat = DCT(f,n=N)

    # print('f_hat_in = ',f_hat_in)
    # print('f        = ',f       )
    # print('f_hat    = ',f_hat   )

    for a,b in zip(np.round(f_hat,12),f_hat_in):
        assert a == b

    print('~~~~ Cosine grid space to coefficient space~~~~~~')

    f_hat = DCT(f_in,n=N)
    f     = IDCT(f_hat,n=N) 
    
    for a,b in zip(np.round(f,12),np.round(f_in,12)):
        assert a == b

    return None;

def Test_Sine_Transform_deal(k,N):

    g_hat_in    = np.zeros(N)
    g_hat_in[k] = 3.3

    x     = grid(N)
    g_in  = 3.3*np.sin((k+1)*x)

    # %%
    print('~~~~ Sine coefficient space to grid space~~~~~~')

    g    = IDST(g_hat_in,n=(3*N)//2)
    g_hat=  DST(g,n=N)

    for a,b in zip(np.round(g_hat,12),g_hat_in):
        assert a == b
    # print('g_hat = ',g_hat_in)
    # print('g     = ',g       )
    # print('g_hat = ',g_hat   )

    print('~~~~ Sine grid space to coefficient space~~~~~~')

    g_hat=  DST(g_in,n=N)
    g    = IDST(g_hat,n=N)

    for a,b in zip(np.round(g,12),np.round(g_in,12)):
        assert a == b

    # print('g     = ',g_in )
    # print('g_hat = ',g_hat)
    # print('g     = ',g    )

    return None;

# Test for 2D data

if __name__ == "__main__":

    # 1D Data
    N = 10;
    k = 1;
    Test_Cosine_Transform(k,N);


    # for k in range(3):
    #     Test_Cosine_Transform(k,N);
    #     Test_Sine_Transform(k,N);

    # #1D Data + Aliasing
    # for k in range(3):
    #     Test_Cosine_Transform_deal(k,N);
    #     Test_Sine_Transform_deal(k,N);