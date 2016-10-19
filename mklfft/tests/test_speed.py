from __future__ import division
import time
import numpy as np

from mklfft import fft, ifft, rfft, irfft

if __name__ == "__main__":

        print('FFT Test speed:')

        n_iter = 200
        M = 1024
        N = 1024
        n = 1024
        axis = 1
        scrambled = False
        inplace = False
        dtype = np.complex128

        start_time = time.time()
        C = np.zeros((M, n), dtype='complex128')
        A = dtype(np.random.randn(M, N))
        for i in range(n_iter):
            C = np.fft.fft(A, n=n, axis=axis)
        npy_time = time.time() - start_time
        print("  numpy --- %s seconds ---" % (npy_time))

        start_time = time.time()
        C = np.zeros((M, n), dtype='complex128')
        A = dtype(np.random.randn(M, N))
        for i in range(n_iter):
            if inplace:
                fft(A, n=n, axis=axis, out=A, scrambled=scrambled)
            else:
                fft(A, n=n, axis=axis, out=C, scrambled=scrambled)
        mkl_time = time.time() - start_time
        print("  mkl   --- %s seconds ---" % (mkl_time))
        print("  Speed-up %.2f" % (npy_time / mkl_time))

        print('RFFT Test speed:')

        n_iter = 200
        M = 1024
        N = 1024
        n = 1024
        axis = 1
        scrambled = False
        dtype = np.float64

        start_time = time.time()
        C = np.zeros((M, n//2+1), dtype='complex128')
        A = dtype(np.random.randn(M, N))
        for i in range(n_iter):
            C = np.fft.rfft(A, axis=axis)
        npy_time = time.time() - start_time
        print("  numpy --- %s seconds ---" % (npy_time))

        start_time = time.time()
        C = np.zeros((M, n//2+1), dtype='complex128')
        A = dtype(np.random.randn(M, N))
        for i in range(n_iter):
            rfft(A, axis=axis, out=C, scrambled=scrambled)
        mkl_time = time.time() - start_time
        print("  mkl   --- %s seconds ---" % (mkl_time))
        print("  Speed-up %.2f" % (npy_time / mkl_time))

