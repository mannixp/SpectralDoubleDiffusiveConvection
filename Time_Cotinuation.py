"""Perform time-stepping continuation of the periodic branch"""
import numpy as np
from Main import Time_Step
from Plot_Tools import Plot_Time_Step, Cartesian_Plot, Energy, Uradial_plot

if __name__ == "__main__":
    
    # %%
    print('Initilaise')
    #filename = 'TimeStep_Gamma9p85.h5'
    #filename = Time_Step(filename, frame=-1, delta_Ra=0)
    filename = 'TimeStep_0.h5'
    Plot_Time_Step(filename, logscale=True, plotting=True)
    Cartesian_Plot(filename, frame=-1, Include_Base_State=False)
    Energy(filename, frame=-1)

    # %%
    # filename = 'TimeStep_14.h5'
    # frame = -1
    # delta_Ra = 20
    # for i in range(20):
    #     filename = Time_Step(filename, frame, delta_Ra)
    #     Plot_Time_Step(filename, logscale=True, plotting=True)
    #     Energy(filename, frame=-1)

    # # %%
    # filename = 'Branches_d0.31325/Small/.h5'
    # Plot_Time_Step(filename, logscale=True, plotting=True)
    # Cartesian_Plot(filename, frame=-1, Include_Base_State=False)
    # Energy(filename, frame=-1)

# %%
