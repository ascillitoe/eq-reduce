import numpy as np
import equadratures as eq
from copy import deepcopy
import time

def b(x,x_bump):
    t = 2.0
    return (np.sin(np.pi*x**(np.log(0.5)/np.log(x_bump))))**t

def deform_airfoil(airfoil,xs,amps,surfs):
    '''
    Function to deform airfoil and return 50D design vector.
    There are easier ways to apply bump functions, but doing it this round-about away since we require design vector for flowfield approximations.
    '''
    scale = 0.005

    #########################################################
    # construct design vector from user specified xs and amps
    #########################################################
    xs = np.array(xs)
    amps = np.array(amps)

    # find suction and pressure indicies in input arrays
    idx_s = [i for i, x in enumerate(surfs) if x == "s"]
    idx_p = [i for i, x in enumerate(surfs) if x == "p"]

    # init
    amp_s = np.zeros(25)
    amp_p = np.zeros(25)
    x_bumps = np.linspace(0.05,0.9,25)

    # Suction
    if len(idx_s)>0:
        # Remove any repeated bumps
        xs_unique, idx_unique = np.unique(xs[idx_s], return_index=True) 
        amps_tmp = amps[idx_s][idx_unique]
        # Find indices of given bumps
        idx_bumps = [~(np.abs(np.subtract.outer(x_bumps,xs_unique)) > 1e-4).all(1)]
        amp_s[idx_bumps] = amps_tmp

    # Pressure
    if len(idx_p)>0:
        # Remove any repeated bumps
        xs_unique, idx_unique = np.unique(xs[idx_p], return_index=True) 
        amps_tmp = amps[idx_p][idx_unique]
        # Find indices of given bumps
        idx_bumps = [~(np.abs(np.subtract.outer(x_bumps,xs_unique)) > 1e-4).all(1)]
        amp_p[idx_bumps] = amps_tmp

    # Design vector
    design_vec = np.empty(50)
    design_vec[0::2] = amp_p
    design_vec[1::2] = amp_s

    ##########################################
    # Apply bump functions to baseline airfoil
    ##########################################
    airfoil_p = airfoil[0:128]
    airfoil_s = airfoil[128:]

    x_base_s = airfoil_s[:,0]
    y_base_s = airfoil_s[:,1]
    x_base_p = airfoil_p[:,0]
    y_base_p = airfoil_p[:,1]

    # Extract the suction and pressure bump amplitudes (beta's) from the design vector
    beta_p = design_vec[0::2]
    beta_s = design_vec[1::2]

    # Apply the bump functions at each x
    npts = y_base_s.shape[0]
    y_s = deepcopy(y_base_s)
    y_p = deepcopy(y_base_p)
    for j in range(25):
        y_s += beta_s[j]*b(x_base_s,x_bumps[j])
        y_p -= beta_p[j]*b(x_base_p,x_bumps[j])
    
    deformed_airfoil = deepcopy(airfoil)
    deformed_airfoil[:,1] = np.hstack([y_p,y_s])

    return deformed_airfoil, design_vec/scale

def eval_poly(x,lower,upper,coeffs,W):
    mybasis = eq.Basis("total-order")
    param = eq.Parameter(distribution='uniform', lower=lower,upper=upper,order=2)
    newpoly = eq.Poly(param, mybasis, method='least-squares')
    newpoly._set_coefficients(user_defined_coefficients=coeffs)
    u = x @ W
    ypred = newpoly.get_polyfit(u)
    return ypred

def standardise(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    Xmin = np.min(X,axis=0)
    Xmax = np.max(X,axis=0)
    Xtrans = 2.0 * ( (X[:,:]-Xmin)/(Xmax - Xmin) ) - 1.0
    return Xtrans
