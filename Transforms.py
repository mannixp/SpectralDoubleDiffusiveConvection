
import numpy as np
import copy
from scipy.fftpack import dct,dst


def grid(N):

    x  = [np.pi*(.5 + i)/(2.*N) for i in range(N)]

    return np.asarray(x);

# --------------------
# Cosine ransforms
# --------------------

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

# -------------------
# Sine transforms
# -------------------

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



# Transforms I use in the code
def DST():

    return None;

def IDST():

    return None;

def DCT():

    return None;

def IDCT():

    return None;



def Test_Cosine_Transform(k,N):

    f_hat_in    = np.zeros(N)
    f_hat_in[k] = 1

    x     = grid(N)
    Cos   = lambda k : np.cos(k*x)
    f_in  = Cos(k)


    print('~~~~ Cosine coefficient space to grid space~~~~~~')

    f_hat_scaled = sinusoid_to_IDCT(f_hat_in);
    f            = dct(f_hat_scaled,type=3) 

    f_hat_scaled = dct(f,type=2)
    f_hat        = DCT_to_sinusoid(f_hat_scaled)

    #print('f_hat = ',f_hat_in)
    #print('f     = ',f       )
    #print('f_hat = ',f_hat   )

    for a,b in zip(np.round(f_hat,12),f_hat_in):
        #print(a,b)
        assert a == b

    print('~~~~ Cosine grid space to coefficient space~~~~~~')

    f_hat_scaled = dct(f_in,type=2)
    f_hat        = DCT_to_sinusoid(f_hat_scaled)

    f_hat_scaled = sinusoid_to_IDCT(f_hat);
    f            = dct(f_hat_scaled,type=3) 

    #print('f     = ',f_in )
    #print('f_hat = ',f_hat)
    #print('f     = ',f    )

    for a,b in zip(np.round(f,12),np.round(f_in,12)):
        #print(a,b)    
        assert a == b

    return None;

def Test_Sine_Transform(k,N):

    g_hat_in    = np.zeros(N)
    g_hat_in[k] = 1

    x     = grid(N)
    Sin   = lambda k : np.sin((k+1)*x)
    g_in  = Sin(k)

    # %%
    print('~~~~ Sine coefficient space to grid space~~~~~~')

    g_hat_scaled = sinusoid_to_IDST(g_hat_in)
    g            = dst(g_hat_scaled,type=3) 

    g_hat_scaled = dst(g,type=2)
    g_hat        = DST_to_sinusoid(g_hat_scaled)

    # print('g_hat = ',g_hat_in)
    # print('g     = ',g       )
    # print('g_hat = ',g_hat   )

    for a,b in zip(np.round(g_hat,12),g_hat_in):
        #print(a,b)
        assert a == b

    print('~~~~ Sine grid space to coefficient space~~~~~~')

    g_hat_scaled = dst(g_in,type=2)
    g_hat        = DST_to_sinusoid(g_hat_scaled)


    g_hat_scaled = sinusoid_to_IDST(g_hat)
    g            = dst(g_hat_scaled,type=3) 

    # print('g     = ',g_in )
    # print('g_hat = ',g_hat)
    # print('g     = ',g    )

    for a,b in zip(np.round(g,12),np.round(g_in,12)):
        #print(a,b)
        assert a == b

    return None;


if __name__ == "__main__":

    k = 1;
    N = 10;

    for k in range(5):
        Test_Cosine_Transform(k,N);

    for k in range(5):
        Test_Sine_Transform(k,N);

