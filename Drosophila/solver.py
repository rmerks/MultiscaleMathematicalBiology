from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

class _Parameters:
    def __init__(self, N, M, R, T, m, h, D, lamb):
        self._check(N, M, R, T, m, h, D, lamb)
        self.N = N
        self.M = M
        self.R = R
        self.T = T
        self.m = m
        self.h = h
        self.D = D
        self.lamb = lamb

    def _check(self, N, M, R, T, m, h, D, lamb):
        assert isinstance(
            R, np.ndarray), f"Parameter is not an array but a {type(R)}. You might have typed R = [...] instead of R[:] = [...]"
        assert isinstance(
            T, np.ndarray), f"Parameter is not an array but a {type(T)}. You might have typed T = [...] instead of T[:] = [...]"
        assert isinstance(
            m, np.ndarray), f"Parameter is not an array but a {type(m)}. You might have typed m = [...] instead of m[:] = [...]"
        assert isinstance(
            h, np.ndarray), f"Parameter is not an array but a {type(h)}. You might have typed h = [...] instead of h[:] = [...]"
        assert isinstance(
            D, np.ndarray), f"Parameter is not an array but a {type(D)}. You might have typed D = [...] instead of D[:] = [...]"
        assert isinstance(
            lamb, np.ndarray), f"Parameter is not an array but a {type(lamb)}. You might have typed lamb = [...] instead of lamb[:] = [...]"

        assert R.shape == (M,)
        assert T.shape == (M, M)
        assert m.shape == (M,)
        assert h.shape == (M,)
        assert D.shape == (M,)
        assert lamb.shape == (M,)

class _ArtistPictures:
    def __init__(self, data, titles, labels, options=None):
        self._data = data
        self._labels = labels
        self._options = options
        self._titles = titles

    def animate(self, outputfile, speed=None):
        filename, extension = Path(outputfile).name.split('.')
        path = Path(outputfile).parent

        

        x, matrices = self._data

        total =int(np.ceil( np.log(len(matrices)) / np.log(10)))

        for k, (matrix, title) in enumerate(zip(matrices, self._titles)):
            plt.figure()
            lineObjects = plt.plot(x, matrix)
            if self._labels:
                plt.legend(lineObjects, self._labels)
            plt.title(title)
            outname = path / (str(filename) + str(k).zfill(total) +'.' +  extension)
            plt.savefig(outname, dpi=300)
            plt.close()

class _ArtistAnimator:
    def __init__(self, data, titles, labels, options=None):
        self._data = data
        self._labels = labels
        self._options = options
        self._titles = titles
        pass

    def animate(self, outputfile, speed=40):
        fig, ax = plt.subplots()
        xaxis = self._data[0]

        matrices = self._data[1]
        matrix = matrices[0]

        lines = ax.plot(xaxis, matrix)

        if self._options:
            ax.set(**self._options)
        if self._labels:
            ax.legend(lines, self._labels)

        def _update(frame):
            # for each frame, update the data stored on each artist.
            #matrix = next(matrices_iter)
            matrix = matrices[frame]
            maximum_value = np.max(matrix) * 1.1
            for k, line in enumerate(lines):
                line.set_ydata(matrix[:, k])
            ax.set_ylim([0, maximum_value])
            ax.set_title(self._titles[frame])
            return lines

        ani = animation.FuncAnimation(
            fig=fig, func=_update, frames=len(matrices), interval=speed)
        ani.save(outputfile)


class Solver:
    def __init__(self, N, M, R, T, m, h, D, lamb, names=None, bicoid_diffusion=None, noflux=False):
        self._par = _Parameters(N, M, R, T, m, h, D, lamb)
        self._n = N * M
        self._dtype = np.float64
        self._prefactor = 10

        self._artist = _ArtistAnimator

        if not bicoid_diffusion:
            D = -np.log(0.5) / 6
        else:
            D = bicoid_diffusion

        self._bicoid_in_nulcei = self._prefactor * \
            np.exp(-D * np.linspace(0, N, N))
        self._laplace = np.diag(
            np.ones(N)*(-2)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)

        if noflux:
            self._laplace[0, 0] = 1  # no flux
            self._laplace[-1, -1] = 1  # no flux

        if names:
            assert len(
                names) == M, "Number of names doesn't correspond with number of protines"
            self._names = names
        else:
            self._names = None

    def _sigmoid(self, u):
        return 0.5 * (u / np.sqrt(u**2 + 1) + 1)

    def _interactions(self, y_matrix):
        par = self._par
        alpha = (par.T @ y_matrix.transpose()).transpose()
        bicoid_contribution = self._bicoid_in_nulcei.reshape(
            (par.N, 1)) @ par.m.reshape((1, par.M))
        return par.R*self._sigmoid(alpha + bicoid_contribution + par.h)

    def _diffusion(self, y_matrix):
        par = self._par
        return (self._laplace @ y_matrix) * par.D

    def _degradation(self, y_matrix):
        return -self._par.lamb * y_matrix

    def _derivative(self, _, y):
        """
        The function calculates d v[i,a] / dt. 
        """
        par = self._par
        # Not a copy, but returns a view of y.
        y_matrix = y.reshape((par.N, par.M))
        output = np.zeros((par.N, par.M), dtype=self._dtype)

        output += self._interactions(y_matrix)
        output += self._diffusion(y_matrix)
        output += self._degradation(y_matrix)

        return output.reshape((self._n,))

    def solve(self, t, y0):
        solution = solve_ivp(self._derivative, t_span=[
                             0, t], y0=self._prefactor*y0, dense_output=False)
        return solution.t, [solution.y[:, k].reshape((self._par.N, self._par.M)) for k, _ in enumerate(solution.t)]

    def solveAndSave(self, t, y0, pathname, filename):
        ts, ys = self.solve(t, y0)

        data = (
            np.array(range(self._par.N)),
            [np.hstack((self._bicoid_in_nulcei.reshape( (self._par.N,1)), y)) for y in ys]
        )
        labels = ['bicoid']
        if self._names:
            labels += self._names
        titles = [f"Time = {t}" for t in ts]
        #_ArtistAnimator(data, titles, labels=labels).animate(pathname + '/' + filename)
        #_ArtistPictures(data, titles, labels=labels).animate(pathname + '/' + filename)
        self._artist(data, titles, labels=labels).animate(pathname + '/' + filename)


def main():
    N = 30
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
    y0 = np.random.rand(N*M)
    ts, ys = Solver(N, M, R, T, m, h, D, lamb).solve(10, y0)

    for k, t in enumerate(ts):
        plt.figure()
        plt.plot(range(N), ys[k])
        plt.title(f"Time = {t}")
        plt.savefig("./out" + f"{k}".zfill(7) + ".png", dpi=300)
        plt.show()


if __name__ == '__main__':
    main()
