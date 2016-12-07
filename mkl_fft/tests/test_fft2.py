from __future__ import division
import time
import numpy as np

from mkl_fft import fft2, ifft2

from unittest import TestCase

class TestMKLFFT2(TestCase):

    def test_fft2(self):

        n_tests = 100
        d_max = 100
        ndim = 2

        fft = dict(forward=dict(mkl=fft2, numpy=np.fft.fft2), backward=dict(mkl=ifft2, numpy=np.fft.ifft2))
        norms = [None, 'ortho']
        orders = ['C', 'F']
        dtypes = [np.float64, np.float32]

        for norm in norms:
            for order in orders:
                for dtype in dtypes:
                    for is_inplace in [True, False]:
                        for direction in fft.keys():
                        
                            for i in range(n_tests):

                                dims = np.random.randint(1,high=d_max+1,size=(ndim))

                                xr = np.array(np.random.normal(size=dims), dtype=dtype)
                                xi = np.array(np.random.normal(size=dims), dtype=dtype)
                                x = np.array(xr + 1j * xi, order=order)

                                if is_inplace:
                                    X = x.copy()
                                    fft[direction]['mkl'](X, norm=norm, out=X)
                                else:
                                    X = fft[direction]['mkl'](x, norm=norm)

                                Y = fft[direction]['numpy'](x, norm=norm)

                                if not np.allclose(X,Y, rtol=1e-2):
                                    print(x)
                                    print(X)
                                    print(Y)
                                    print("Error {}".format(np.linalg.norm(X - Y)))
                                    params = 'dims={} norm={} in_place={} order={} dtype={}'.format(dims, norm, is_inplace, order, dtype)
                                    print('  Failed fft2 {} with {}'.format(direction, params))
                                    passed = False
                                else:
                                    passed = True

                                self.assertTrue(passed)
                            
