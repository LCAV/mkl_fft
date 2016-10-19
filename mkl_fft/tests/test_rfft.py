from __future__ import division
import time
import numpy as np
#import mklfft as fft
from mkl_fft import rfft, irfft

from unittest import TestCase

class TestMKLRFFT(TestCase):

    def test_rfft(self):
        n_tests = 1000
        n_max = 2
        d_max = 100

        norm = 'ortho'
        inplace = True
        scrambled = True
        dtype = np.float64

        passed = True

        for i in range(n_tests):
            ndim = np.random.randint(1,high=n_max+1)
            axis = np.random.randint(0,high=ndim)
            dims = np.random.randint(1,high=d_max+1,size=(ndim))

            x = np.array(np.random.normal(size=dims), dtype=dtype)
            x = np.asfortranarray(x)

            X = rfft(x, axis=axis, norm=norm, scrambled=scrambled)

            Y = np.fft.rfft(x, axis=axis, norm=norm)

            x_ = irfft(X, n=x.shape[axis], axis=axis, norm=norm, scrambled=scrambled)

            if not np.allclose(X,Y):
                print('  Failed forward with ndim=%d axis=%d dims=' % (ndim, axis), dims)
                passed = False

            if not np.allclose(x, x_):
                print('  Failed backward with ndim=%d axis=%d dims=' % (ndim, axis), dims)
                passed = False

        self.assertTrue(passed)


