from __future__ import division
import time
import numpy as np

from mkl_fft import fft, ifft

from unittest import TestCase

class TestMKLFFT(TestCase):

    def test_fft(self):

        n_tests = 1000
        n_max = 2
        d_max = 100

        norm = 'ortho'
        inplace = True
        scrambled = True

        passed = True

        for i in range(n_tests):
            ndim = np.random.randint(1,high=n_max+1)
            axis = np.random.randint(0,high=ndim)
            dims = np.random.randint(1,high=d_max+1,size=(ndim))

            x = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)

            if inplace:
                X = x.copy()
                fft(X, axis=axis, norm=norm, out=X, scrambled=scrambled)
            else:
                X = fft(x, axis=axis, norm=norm, scrambled=scrambled)

            Y = np.fft.fft(x, axis=axis, norm=norm)

            x_ = ifft(X, axis=axis, norm=norm)

            if not np.allclose(X,Y) and not scrambled:
                print('  Failed forward with ndim=%d axis=%d dims=' % (ndim, axis), dims)
                passed = False

            if not np.allclose(x, x_):
                print('  Failed backward with ndim=%d axis=%d dims=' % (ndim, axis), dims)
                passed = False

        self.assertTrue(passed)
