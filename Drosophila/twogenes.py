from solver import Solver
import numpy as np


def _make_solver():
    N = 30
    M = 2
    dtype = np.float64
    
    R = np.zeros((M,), dtype=dtype)
    T = np.zeros((M, M), dtype=dtype)
    m = np.zeros((M,), dtype=dtype)
    h = np.zeros((M,), dtype=dtype)
    D = np.zeros((M,), dtype=dtype)
    lamb = np.zeros((M,), dtype=dtype)
    
    
    T[0,:] = [1, -1]
    T[1,:] = [-1, 1]
    
    m[:] = [0, 0]
    h[:] = [0, 0]
    R[:] = [1,1]
    lamb[:] = [0.01,0.01]
    D[:] = [0, 0]
    
    return Solver(N, M, R, T,m,h,D,lamb)

solver = _make_solver()


if __name__ == '__main__':
    y0 = np.random.rand(30*2)
    solver.solveAndSave2(1000, y0, "./twogenes", "out.mp4")

