import numpy as np



N = 3
M = 2
dtype = np.float64

R = np.zeros( (M,) ,dtype=dtype)
T = np.zeros( (M,M),dtype=dtype )
m = np.zeros( (M,) ,dtype=dtype)
h = np.zeros( (M,) ,dtype=dtype)
D = np.zeros( (M,) ,dtype=dtype)
lamb = np.zeros( (M,), dtype=dtype )

R[:] = [1,2]
T[0,:] = [3,4]
T[1,:] = [5,6]
m[:] = [7,8]
D[:] = [0.1, 0.2]
h[:] = [9, 10]
lamb[:] = [0.4, 0.5]

v = np.array([10, 11,12,13,14,15]).reshape( (N,M))

def g(u):
    return 0.5 * ( u / np.sqrt(u**2 + 1) + 1)

vbic = [1, 1/2, 0]

from solver import Solver
solver = Solver(N, M, R, T, m, h, D, lamb)

def f(a,i):

    print("-"*50)
    print(f"Calculating for {a,i=}")
    output = R[a] * g( T[a,0] * v[i,0] + T[a,1] * v[i,1] + m[a]*vbic[a] + h[a] )  
    print(f" % interactions = {output}")
    print(f" % interactions from solver = {solver._interactions(v)[i,a]}")


    diffusion = -D[a] * 2*v[i,a] 
    if i-1 > 0:
        diffusion += D[a]* ( v[i-1, a] )
    if i+1 < N:
        diffusion += D[a]* v[i+1, a] 
    print(f" % diffusions = {diffusion}")
    print(f" % diffusions from solver = {solver._diffusion(v)[i,a]}")

    degradation = - lamb[a] * v[i,a]
    print(f" % degradation = {degradation}")
    print(f" % degradation from solver = {solver._degradation(v)[i,a]}")
    output += degradation
        
    print(f" % total = {output}")
    print("-"*50)



for a in range(2):
    for i in range(3):
        f(a-1,i-1)


