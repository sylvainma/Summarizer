import numpy as np
import cython
cimport numpy as cnp
from tqdm import tqdm

def calc_scatters(cnp.ndarray K):
    """
    Calculate scatter matrix:
    scatters[i,j] = {scatter of the sequence with starting frame i and ending frame j} 
    """
    cdef:
        int i, j, n
        cnp.ndarray K1, K2, scatters
        
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1); # TODO: use the fact that K - symmetric

    scatters = np.zeros((n, n));

    for i in tqdm(range(n), desc="scatters"):
        for j in range(i, n):
            scatters[i, j] = K1[j+1] - K1[i] \
                - (K2[j+1, j+1] + K2[i, i] - K2[j+1, i] - K2[i, j+1]) / (j - i + 1)
    
    return scatters

def cpd_nonlin(cnp.ndarray K, int ncp, int lmin=1, int lmax=100000, backtrack=True, verbose=True,
    out_scatters=None):

    """ Change point detection with dynamic programming
    K - square kernel matrix 
    ncp - number of change points to detect (ncp >= 0)
    lmin - minimal length of a segment
    lmax - maximal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)
    
    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )    
        obj_vals - values of the objective function for 0..m changepoints
        
    """
    cdef:
        int m, n, n1, k, l, t
        double c
        cnp.ndarray I, J, p
        
    m = int(ncp)  # prevent numpy.int64

    n, n1 = K.shape[0], K.shape[1]
    assert(n == n1), "Kernel matrix awaited."    
    
    assert(n >= (m + 1)*lmin)
    assert(n <= (m + 1)*lmax)
    assert(lmax >= lmin >= 1)
    
    if verbose:
        #print "n =", n
        print( "Precomputing scatters...")
    J = calc_scatters(K)
    
    if out_scatters != None:
        out_scatters[0] = J

    if verbose:
        print( "Inferring best change points...")
    # I[k, l] - value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t[k] when t[k+1] equals l
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)

    for k in tqdm(range(1, m+1), desc="cpd_nonlin loop"):
        for l in range((k+1) * lmin, n+1):
            I[k, l] = 1e100
            for t in range(max(k * lmin, l - lmax), l - lmin + 1):
                c = I[k-1, t] + J[t, l-1]
                if c < I[k, l]:
                    I[k, l] = c
                    if backtrack:
                        p[k, l] = t
    
    # Collect change points
    cps = np.zeros(m, dtype=int)
    
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy() 
    scores[scores > 1e99] = np.inf
    return cps, scores
    

