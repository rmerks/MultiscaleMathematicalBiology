from solver import Solver
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
    
    
    T[0,:] = [0,0,0,0,0]
    T[1,:] = [0,0,0,0,0]
    T[2,:] = [0,0,0,0,0]
    T[3,:] = [0,0,0,0,0]
    T[4,:] = [0,0,0,0,0]
    
    h[:] = [0.1, 0.1 ,0.1, 0.1, 0]
    R[:] = [1, 1, 1, 1, 1]
    lamb[:] = [5, 5, 5, 5, 5]
    D[:] = [1,1,1,1,1]
    
    
    return Solver(N, M, R, T,m,h,D,lamb)

solver = _make_solver()


if __name__ == '__main__':
    y0 = np.random.rand(30*5)
    solver.solveAndSave(10, y0, "./", "dros")

