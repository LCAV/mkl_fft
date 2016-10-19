from __future__ import division, print_function
import time
import numpy as np

from mkl_fft import fft, ifft, rfft, irfft, fft2

if __name__ == "__main__":

    import time

    n_iter = 200
    M = 1
    N = 2**14
    sqrtN = int(np.sqrt(N))

    scrambled=False

    np.seterr(all='raise')

    algos = {
            'fft2': { 
                'Numpy': np.fft.fft2, 
                'MKL': fft2,
                'shape': { 'in': (sqrtN, sqrtN), 'out': (sqrtN, sqrtN) },
                'single': { 'in': np.complex64, 'out': np.complex64, },
                'double': { 'in': np.complex128, 'out': np.complex128, },
                },
            'fft': { 
                'Numpy': np.fft.fft, 
                'MKL': fft,
                'shape': { 'in': (M, N,), 'out': (M, N,) },
                'single': { 'in': np.complex64, 'out': np.complex64, },
                'double': { 'in': np.complex128, 'out': np.complex128, },
                },
            'rfft': { 
                'Numpy': np.fft.rfft, 
                'MKL': rfft,
                'shape': { 'in': (M, N,), 'out': (M, N // 2 + 1,) },
                'single': { 'in': np.float32, 'out': np.complex64, },
                'double': { 'in': np.float64, 'out': np.complex128, },
                },
            }

    for algo in ['fft', 'fft2', 'rfft']:
        for precision in ['single', 'double']:

            A = algos[algo][precision]['in'](np.random.randn(*(algos[algo]['shape']['in'])))

            start_time = time.time()
            for i in range(n_iter):
                C = algos[algo]['Numpy'](A)
            npy_time = time.time() - start_time

            A = algos[algo][precision]['in'](np.random.randn(*(algos[algo]['shape']['in'])))
            C = np.zeros(algos[algo]['shape']['out'], dtype=algos[algo][precision]['out'])

            start_time = time.time()
            for i in range(n_iter):
                #algos[algo]['MKL'](A, out=C)
                if algo != 'fft2':
                    C = algos[algo]['MKL'](A, scrambled=scrambled)
                else:
                    C = algos[algo]['MKL'](A)
            mkl_time = time.time() - start_time

            print("---",algo,precision,":")
            print("--- NPY: %s seconds ---" % npy_time)
            print("--- MKL: %s seconds ---" % mkl_time)
            print("    Speed-up %.2f" % (npy_time / mkl_time))
            print()


