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


# Regional Drug Receptor Modulation (RDRM) constants for their transfer functions:
# --------------------------------------------------------------------------
Receptor = 0
wgaini=0
wgaine=0


# transfer functions:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=310;
be=125;
de=0.16;
def phie(x):
    y = (ae*x-be)*(1+Receptor*wgaine) #  for LSD
    # if (y != 0):
    return y/(1-np.exp(-de*y))
    # else:
    #     return 0


# transfer function: inhibitory
ai=615;
bi=177;
di=0.087;
def phii(x):
    y = (ai*x-bi)*(1+Receptor*wgaini) # for LSD
    # if (y != 0):
    return y/(1-np.exp(-di*y))
    # else:
    #     return 0

