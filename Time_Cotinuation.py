"""Perform time-stepping continuation of the periodic branch"""
import numpy as np
from Main import Time_Step
from Plot_Tools import Plot_Time_Step, Cartesian_Plot, Energy, Uradial_plot

if __name__ == "__main__":
    
    # %%
    print('Initilaise')
    # filename = 'TimeStep_0.h5'
    # filename = Time_Step(filename, frame=-1, delta_Ra=0.33, Ra_new=4000)
    
    # Plot_Time_Step(filename, logscale=True, plotting=True)
    # Cartesian_Plot(filename, frame=-1, Include_Base_State=False)
    # Energy(filename, frame=-1)

    # %%
    for d_new in [0.33, 0.325, 0.32, 0.315, 0.31, 0.305, 0.3]:
        print('d_new = %3.3f \n'%d_new)
        filename = Time_Step(open_filename='TimeStep_10.h5', frame=-1, d_new=d_new)
        Plot_Time_Step(filename, logscale=True, plotting=True)
        Cartesian_Plot(filename, frame=-1, Include_Base_State=False)
        Energy(filename, frame=-1)

    # %%
    filename = 'TimeStep_10.h5'
    Plot_Time_Step(filename, logscale=True, plotting=True)
    Cartesian_Plot(filename, frame=-1, Include_Base_State=False)
    Energy(filename, frame=-1)

# %%
