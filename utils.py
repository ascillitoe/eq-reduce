import numpy as np
import scipy as sp
import equadratures as eq
from copy import deepcopy
import time
import urllib.parse
import re

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

def get_airfoils(base_airfoil,Xsamples):
    nsamples = Xsamples.shape[0]
    scale = 0.005 #factor to rescale x vectors from eq by
    newscale = 2.0 #factor to scale new designs' delta's by (for easy viz)
    # Split airfoil coords into pressure and suction
    base_airfoil_p = base_airfoil[0:128]
    base_airfoil_s = base_airfoil[128:]

    # x locations of bumps (same for suction and pressure surfaces)
    x_bump = np.linspace(0.05,0.9,25)

    # Loop through all the sample designs
    npts = base_airfoil_s.shape[0]
    y_s = np.empty([npts,nsamples])
    y_p = np.empty([npts,nsamples])
    for samp, X_design in enumerate(Xsamples):
        # Extract the suction and pressure bump amplitudes (beta's) from the design vector
        beta_p = scale*X_design[0::2]
        beta_s = scale*X_design[1::2]
    
        # Apply the bump functions at each x
        y_s_tmp = deepcopy(base_airfoil_s[:,1])
        y_p_tmp = deepcopy(base_airfoil_p[:,1])
        for j in range(25):
            y_s_tmp += newscale*beta_s[j]*b(base_airfoil_s[:,0],x_bump[j])
            y_p_tmp -= newscale*beta_p[j]*b(base_airfoil_p[:,0],x_bump[j])
    
        y_s[:,samp] = y_s_tmp
        y_p[:,samp] = y_p_tmp

    return y_p, y_s

def eval_poly(x,lower,upper,coeffs,W):
    mybasis = eq.Basis("total-order")
    param = eq.Parameter(distribution='uniform', lower=lower,upper=upper,order=2)
    newpoly = eq.Poly(param, mybasis, method='least-squares')
    newpoly._set_coefficients(user_defined_coefficients=coeffs)
    u = x @ W
    ypred = newpoly.get_polyfit(u)
    return ypred

#########################################################
# NOTE - All methods below are adapted from equadratures
#########################################################
def standardise(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    Xmin = np.min(X,axis=0)
    Xmax = np.max(X,axis=0)
    Xtrans = 2.0 * ( (X[:,:]-Xmin)/(Xmax - Xmin) ) - 1.0
    return Xtrans

def null_space(A, rcond=None):
    '''
    null space method adapted from scipy.
    '''
    u, s, vh = sp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def get_samples_constraining_active_coordinates(W, inactive_samples, active_coordinates):
    """

    A hit and run type sampling strategy for generating samples at a given coordinate in the active subspace
    by varying its coordinates along the inactive subspace.

    :param Subspaces self:
        An instance of the Subspaces object.
    :param int inactive_samples:
        The number of inactive samples required.
    :param numpy.ndarray active_coordiantes:
        The active subspace coordinates.

    :return:
        **X**: An numpy.ndarray of the full-space coordinates.

    **Note:**
    This routine has been adapted from Paul Constantine's hit_and_run() function; see reference below.

    Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., Fletcher, L., (2016) Python Active-Subspaces Utility Library. Journal of Open Source Software, 1(5), 79. `Paper <http://joss.theoj.org/papers/10.21105/joss.00079>`__.

    """
    y = active_coordinates
    N = inactive_samples
    W1 = W # active_subspace is provided
    W2 = null_space(W1.T) #inactive subspace
    M = np.hstack([W1,W2])

    m, n = W1.shape
    s = np.dot(W1, y).reshape((m, 1))
    normW2 = np.sqrt(np.sum(np.power(W2, 2), axis=1)).reshape((m, 1))
    A = np.hstack((np.vstack((W2, -W2.copy())), np.vstack((normW2, normW2.copy()))))
    b = np.vstack((1 - s, 1 + s)).reshape((2 * m, 1))
    c = np.zeros((m - n + 1, 1))
    c[-1] = -1.0
    # print()

    zc = linear_program_ineq(c, -A, -b)
    z0 = zc[:-1].reshape((m - n, 1))

    # define the polytope A >= b
    s = np.dot(W1, y).reshape((m, 1))
    A = np.vstack((W2, -W2))
    b = np.vstack((-1 - s, -1 + s)).reshape((2 * m, 1))

    # tolerance
    ztol = 1e-6
    eps0 = ztol / 4.0

    Z = np.zeros((N, m - n))
    for i in range(N):

        # random direction
        bad_dir = True
        count, maxcount = 0, 50
        while bad_dir:
            d = np.random.normal(size=(m - n, 1))
            bad_dir = np.any(np.dot(A, z0 + eps0 * d) <= b)
            count += 1
            if count >= maxcount:
                Z[i:, :] = np.tile(z0, (1, N - i)).transpose()
                yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
                return np.dot(M, yz).T

        # find constraints that impose lower and upper bounds on eps
        f, g = b - np.dot(A, z0), np.dot(A, d)

        # find an upper bound on the step
        min_ind = np.logical_and(g <= 0, f < -np.sqrt(np.finfo(np.float).eps))
        eps_max = np.amin(f[min_ind] / g[min_ind])

        # find a lower bound on the step
        max_ind = np.logical_and(g > 0, f < -np.sqrt(np.finfo(np.float).eps))
        eps_min = np.amax(f[max_ind] / g[max_ind])

        # randomly sample eps
        eps1 = np.random.uniform(eps_min, eps_max)

        # take a step along d
        z1 = z0 + eps1 * d
        Z[i, :] = z1.reshape((m - n,))

        # update temp var
        z0 = z1.copy()

    yz = np.vstack([np.repeat(y[:, np.newaxis], N, axis=1), Z.T])
    return np.dot(M, yz).T

def linear_program_ineq(c, A, b):
    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make unbounded bounds
    bounds = []
    for i in range(c.size):
        bounds.append((None, None))

    A_ub, b_ub = -A, -b
    res = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options={"disp": False}, method='simplex')
    if res.success:
        return res.x.reshape((c.size, 1))
    else:
        np.savez('bad_scipy_lp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                 c=c, A=A, b=b, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')


def airfoil_mask(xx,yy,airfoil_x,airfoil_y):
    tol = 5e-3# remove points if within this tolerance of surface
    nx,ny = xx.shape
    airfoil_x_p = airfoil_x[0:128]
    airfoil_y_p = airfoil_y[0:128]
    airfoil_x_s = airfoil_x[128:]
    airfoil_y_s = airfoil_y[128:]
 
    # Find i indices within airfoil range
    x = xx[:,0]
    idx = np.where((x>=0.0) & (x<=1.0))[0]

    # Loop through i indices within airfoil range, set as NaN if "within" airfoil solid region
    for i in idx:
        # Find suction and pressure coords at this given x coord
        y_p = np.interp(x[i],airfoil_x_p[::-1],airfoil_y_p[::-1])
        y_s = np.interp(x[i],airfoil_x_s,airfoil_y_s)

        # For all j at this i, set to NaN if >y_p and <y_s (only need to set yy to nan to hide on scatter plot) TODO - this could be vectorised further for speed...
        y = yy[i,:]
        idx2 = np.where((y>y_p-tol)&(y<y_s+tol))
        yy[i,idx2] = np.nan

    return xx,yy

def convert_latex(text):
    def toimage(x):
        if x[1] and x[-2] == r'$':
            x = x[2:-2]
            img = "\n<img src='https://math.now.sh?from={}&color=black' style='display: block; margin: 0.5em auto;'/>\n"
            return img.format(urllib.parse.quote_plus(x))
        else:
            x = x[1:-1]
            return r'![](https://math.now.sh?inline={}&color=black)'.format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$', lambda x: toimage(x.group()), text)
