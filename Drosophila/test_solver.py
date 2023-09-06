import unittest
import numpy as np

from .solver import Solver, _Parameters


class TestParameter(unittest.TestCase):
    def testCheck_R(self):
        N = 3
        M = 2
        dtype = np.float64

        R = np.zeros((M-1,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        self.assertRaises(AssertionError, lambda: _Parameters(
            N, M, R, T, m, h, D, lamb))

    def testCheck_T(self):
        N = 3
        M = 2
        dtype = np.float64

        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M+1, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        self.assertRaises(AssertionError, lambda: _Parameters(
            N, M, R, T, m, h, D, lamb))

    def testCheck_m(self):
        N = 3
        M = 2
        dtype = np.float64

        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M+10,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        self.assertRaises(AssertionError, lambda: _Parameters(
            N, M, R, T, m, h, D, lamb))

    def testCheck_h(self):
        N = 3
        M = 2
        dtype = np.float64

        R = np.zeros((M-1,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((1,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        self.assertRaises(AssertionError, lambda: _Parameters(
            N, M, R, T, m, h, D, lamb))


class TestSolver(unittest.TestCase):

    def setUp(self):
        N = 3
        M = 2
        dtype = np.float64

        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        R[:] = [1, 2]
        T[0, :] = [3, 4]
        T[1, :] = [5, 6]
        m[:] = [7, 8]
        D[:] = [0.1, 0.2]
        h[:] = [9, 10]
        lamb[:] = [0.4, 0.5]

        y0 = np.array([10, 11, 12, 13, 14, 15]).reshape((N, M))

        solver = Solver(N, M, R, T, m, h, D, lamb)

        self.solution = solver._derivative(0, y0).reshape((N, M))

    def testV11(self):
        self.assertAlmostEqual(self.solution[0, 0], -3.8, delta=7)

    def testV22(self):
        self.assertAlmostEqual(self.solution[1, 1], -3.8, delta=6)
        print(self.solution)

    def testInteractions(self):

        N = 3
        M = 2
        dtype = np.float64
        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        def g(u):
            return 0.5 * (u / np.sqrt(u**2 + 1) + 1)

        R[:] = [1, 2]
        T[0, :] = [3, 4]
        T[1, :] = [5, 6]
        m[:] = [7, 8]
        D[:] = [0.1, 0.2]
        h[:] = [9, 10]
        lamb[:] = [0.4, 0.5]

        vbic = [1, 1/2, 0]
        vbic = np.exp(-np.linspace(0, 1, N))

        v = np.array([10, 11, 12, 13, 14, 15]).reshape((N, M))
        solver = Solver(N, M, R, T, m, h, D, lamb)

        for a in range(M):
            for i in range(N):
                byhand = R[a] * g(T[a, 0] * v[i, 0] + T[a, 1]
                                 * v[i, 1] + m[a]*vbic[a] + h[a])
                bysolver = solver._interactions(v)[i, a]
                self.assertAlmostEqual(byhand, bysolver, delta=3)

    def testDiffusion(self):

        N = 3
        M = 2
        dtype = np.float64
        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        def g(u):
            return 0.5 * (u / np.sqrt(u**2 + 1) + 1)

        R[:] = [1, 2]
        T[0, :] = [3, 4]
        T[1, :] = [5, 6]
        m[:] = [7, 8]
        D[:] = [0.1, 0.2]
        h[:] = [9, 10]
        lamb[:] = [0.4, 0.5]

        vbic = [1, 1/2, 0]
        vbic = np.exp(-np.linspace(0, 1, N))

        solver = Solver(N, M, R, T, m, h, D, lamb)
        v = np.array([10, 11, 12, 13, 14, 15]).reshape((N, M))
        for a in range(M):
            for i in range(N):

                diffusion = -D[a] * 2*v[i, a]
                if i-1 > 0:
                    diffusion += D[a] * (v[i-1, a])
                if i+1 < N:
                    diffusion += D[a] * v[i+1, a]
                byhand = diffusion
                bysolver = solver._diffusion(v)[i, a]
                self.assertAlmostEqual(byhand, bysolver, delta=3)

    def testDegradation(self):

        N = 3
        M = 2
        dtype = np.float64
        R = np.zeros((M,), dtype=dtype)
        T = np.zeros((M, M), dtype=dtype)
        m = np.zeros((M,), dtype=dtype)
        h = np.zeros((M,), dtype=dtype)
        D = np.zeros((M,), dtype=dtype)
        lamb = np.zeros((M,), dtype=dtype)

        def g(u):
            return 0.5 * (u / np.sqrt(u**2 + 1) + 1)

        R[:] = [1, 2]
        T[0, :] = [3, 4]
        T[1, :] = [5, 6]
        m[:] = [7, 8]
        D[:] = [0.1, 0.2]
        h[:] = [9, 10]
        lamb[:] = [0.4, 0.5]

        vbic = [1, 1/2, 0]
        solver = Solver(N, M, R, T, m, h, D, lamb)

        v = np.array([10, 11, 12, 13, 14, 15]).reshape((N, M))
        for a in range(M):
            for i in range(N):
                byhand = - lamb[a] * v[i, a]
                bysolver = solver._degradation(v)[i, a]
                self.assertAlmostEqual(byhand, bysolver, delta=3)


if __name__ == '__main__':
    unittest.main()
