# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Euler-Maruyama Integrator
# Based on the code from the paper:
#
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from functions.randn2 import randn2
from functions.Utils import numTricks as iC
from numba import jit

print("Going to use the Euler-Maruyama Integrator...")

neuronalModel = None  # To be able to choose the model externally...
verbose = True


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    neuronalModel.recompileSignatures()
    recordBookkeeping.recompile()
    integrationStep.recompile()
    # pass

# Functions to convert the stimulus from a function to an array
# --------------------------------------------------------------------------
stimuli = None  # To add some stimuli, if needed...
allStimuli = None
def initStimuli(dt, Tmaxneuronal):
    global allStimuli
    tValues = np.arange(0, Tmaxneuronal, dt)
    if stimuli:
        allStimuli = np.array(list(map(stimuli.stimulus, tValues)))
    else:
        allStimuli = np.zeros(len(tValues))


# Matlab compatible definitions
# --------------------------------------------------------------------------
MatlabCompatibility = False
if MatlabCompatibility:
    def randn(vars,N):
        ra = randn2(vars,N)
        return ra #.reshape(-1)
else:
    from numpy.random import randn


# # bookkeeping vars & methods -> Just forward them to the neuronal model we are using...
# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
# --------------------------------------------------------------------------
ds = 1  # downsampling stepsize
def initBookkeeping(N, tmax):
    # global curr_xn, curr_rn, nn
    # global curr_obsVars
    # curr_xn = np.zeros((int(tmax), N))
    # curr_rn = np.zeros((int(tmax), N))
    obsVars = neuronalModel.numObsVars()
    timeElements = int(tmax/ds) + 1  # the last +1 because of isClose roundings...
    return np.zeros((timeElements, obsVars, N))


@jit(nopython=True)
def recordBookkeeping(t, obsVars, curr_obsVars):
    # global curr_obsVars
    if iC.isInt(t/ds):
        nn = int(np.round(t/ds))  # is is an int-ish...
        curr_obsVars[nn,:,:] = obsVars[:,:]
    return curr_obsVars


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Euler-Maruyama Integration
# --------------------------------------------------------------------------
sigma = 0.01
clamping = True
@jit(nopython=True)
def integrationStep(simVars, dt, stimulus):  #, curr_obsVars, doBookkeeping):
    numSimVars = simVars.shape[0]; N = simVars.shape[1]
    dvars_obsVars = neuronalModel.dfun(simVars, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    simVars = simVars + dt * dvars + np.sqrt(dt) * sigma * randn(numSimVars,N)  # Euler-Maruyama integration.
    if clamping:
        simVars = np.where(simVars > 1., 1., simVars)  # clamp values to 0..1
        simVars = np.where(simVars < 0., 0., simVars)
    return simVars, obsVars


# # @jit(nopython=True)
def integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars):
    for t in np.arange(0, Tmaxneuronal, dt):
        stimulus = allStimuli[int(t / dt)]
        simVars_obsVars = integrationStep(simVars, dt, stimulus)
        simVars = simVars_obsVars[0]; obsVars = simVars_obsVars[1]  # cannot use unpacking in numba...
        if doBookkeeping:
            curr_obsVars = recordBookkeeping(t, obsVars, curr_obsVars)
    return simVars, curr_obsVars


# # @jit(nopython=True)
def integrate(dt, Tmaxneuronal, simVars, doBookkeeping = True):
    # numSimVars = simVars.shape[0]
    N = simVars.shape[1]  # N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    curr_obsVars = initBookkeeping(N, Tmaxneuronal)
    return integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars)


# ==========================================================================
# ==========================================================================
# ==========================================================================
def simulate(dt, Tmaxneuronal):
    if verbose:
        print("Simulating...", flush=True)
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars)
    return obsVars


def warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=10000):
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    if verbose:
        print("Warming Up...", end=" ", flush=True)
    initStimuli(dt, TWarmUp)
    simVars, obsVars = integrate(dt, TWarmUp, simVars, doBookkeeping=False)
    if verbose:
        print("and simulating!!!", flush=True)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars, doBookkeeping=True)
    return obsVars


# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
