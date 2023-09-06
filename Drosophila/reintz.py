from .solver import Solver
import numpy as np


def _make_solver():
    N = 30
    M = 5
    dtype = np.float64
    
    R = np.zeros((M,), dtype=dtype)
    T = np.zeros((M, M), dtype=dtype)
    m = np.zeros((M,), dtype=dtype)
    h = np.zeros((M,), dtype=dtype)
    D = np.zeros((M,), dtype=dtype)
    lamb = np.zeros((M,), dtype=dtype)
    
    
    T[0,:] = [0.34, -0.86, -0.44, 1.6, 0.005, ]
    T[1,:] = [0.013, 0.11, -0.9, -0.23, 0.002 ]
    T[2,:] = [-2.5, -0.28, 0.15, -0.24, 0.001]
    T[3,:] = [-0.33, -4.31, -1.6, -0.076, 0.003]
    T[4,:] = [-2.3, -3, -1.9, -1.4, 0.039]
    
    m[:] = [1.8, 7.6, 1.1, -1.8, 14]
    h[:] = [-1.5, 18, 0.59, -4.5, 4.4]
    R[:] = [0.49, 0.96, 1.1, 1.1, 0.8]
    halftimes = [19, 8.2, 7.6, 8.7, 6] # in the paper the halftimes are reported.

    lamb[:] = [np.log(2) / htime for htime in halftimes] # The program uses decayrates. The relation is log(2) / decayrate = halftime.
    D[:] = [0.18, 0.00037, 0.07,0.03,7.6e-5]
    
    names = ['kr', 'hb', 'gt', 'kni','eve'] 
    return Solver(N, M, R, T,m,h,D,lamb, names=names)

solver = _make_solver()


if __name__ == '__main__':
    y0 = np.random.rand(30*5)
    solver.solveAndSave(100, y0, "./", "reintz.png")

