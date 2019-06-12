# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#   
#  Computes simulations with the Dynamic Mean Field Model (DMF) using 
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition 
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#
#   Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#   Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., 
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
from pathlib import Path
from functions import Balance_J9
from functions import DynamicMeanField, serotonin2A
from functions import simulateFCD


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.Inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


# Load Structural Connectivity Matrix
sc90 = sio.loadmat('all_SC_FC_TC_76_90_116.mat')['sc90']  #load LSDnew.mat tc_aal
C=sc90/np.max(sc90[:])*0.2

NumSubjects = 15  # Number of Subjects in empirical fMRI dataset


# Setup for Serotonin 2A-based DMF simulation!!!
# ----------------------------------------------
DynamicMeanField.He = serotonin2A.phie
DynamicMeanField.Hi = serotonin2A.phii

# Load Regional Drug Receptor Map
mean5HT2A_aalsymm = sio.loadmat('mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).reshape((-1,1))


# ============================================================================
# ============================================================================
# ============================================================================
# Define optimal parameters
we = 2.1  # Global Coupling parameter
serotonin2A.wgaini = 0  # Placebo conditions, to calibrate the J's...
serotonin2A.wgaine = 0
# ==== J is calculated this only once, then saved
if not Path("J_Balance.mat").is_file():
    print("Computing J_Balance !!!")
    DynamicMeanField.J=Balance_J9.JOptim(we,C) # This is the Feedback Inhibitory Control
    sio.savemat('J_Balance.mat', {'J': J})  # save J_Balance J
else:
    print("Loading J_Balance !!!")
    # ==== J can be calculated only once and then load J_Balance J
    DynamicMeanField.J = sio.loadmat('J_Balance.mat')['J']


# ============================================================================
# ============================================================================
# ============================================================================
if not Path("FCD_values_placebo.mat").is_file():
    # SIMULATION OF OPTIMAL PLACEBO
    print("SIMULATION OF OPTIMAL PLACEBO")
    wge = 0 # 0 for placebo, 0.2 for LSD
    serotonin2A.wgaini = 0
    serotonin2A.wgaine = wge

    np.random.seed(42)

    cotsampling_pla_s = simulateFCD.simulate(NumSubjects, C, we)

    sio.savemat('FCD_values_placebo.mat', {'cotsampling_pla_s': cotsampling_pla_s})  # save FCD_values_placebo cotsampling_pla_s
else:
    print("LOADING OPTIMAL PLACEBO")
    cotsampling_pla_s = sio.loadmat('FCD_values_placebo.mat')['cotsampling_pla_s']

# ============================================================================
# ============================================================================
# ============================================================================
if not Path("FCD_values_placebo.mat").is_file():
    # SIMULATION OF OPTIMAL LSD fit
    print("SIMULATION OF OPTIMAL LSD fit ")
    wge = 0.2 # 0 for placebo, 0.2 for LSD
    serotonin2A.wgaini = 0
    serotonin2A.wgaine = wge

    np.random.seed(42)

    cotsampling_lsd_s = simulateFCD.simulate(NumSubjects, C, we)

    sio.savemat('FCD_values_lsd.mat', {'cotsampling_lsd_s': cotsampling_lsd_s})  # save FCD_values_lsd cotsampling_lsd_s
else:
    print("LOADING OPTIMAL LSD fit")
    cotsampling_lsd_s = sio.loadmat('FCD_values_lsd.mat')['cotsampling_lsd_s']


# ============================================================================
# Plot
# ============================================================================
[h_lsd, x] = my_hist(cotsampling_lsd_s[:].T.flatten(), np.arange(-.1, 1.025, .025))
[h_pla, x1] = my_hist(cotsampling_pla_s[:].T.flatten(), np.arange(-.1, 1.025, .025))

import matplotlib.pyplot as plt

width=0.01
plaBar = plt.bar(x, h_pla, width=0.01, color="red", label="Placebo")
lsdBar = plt.bar(x+width, h_lsd, width=0.01, color="blue", label="LSD")
plt.xlabel('FCD values')
plt.ylabel('Count')
plt.legend(handles=[lsdBar, plaBar], loc='upper right')
plt.title('Simulated data')
plt.show()
