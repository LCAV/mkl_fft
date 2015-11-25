from __future__ import division
import time
import numpy as np
import mklfft as fft

n_tests = 1000
n_max = 2
d_max = 100

norm = 'ortho'
inplace = True

passed = True

print 'Test correctness:'
for i in xrange(n_tests):
    ndim = np.random.randint(1,high=n_max+1)
    axis = np.random.randint(0,high=ndim)
    dims = np.random.randint(1,high=d_max+1,size=(ndim))

    x = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)

    if inplace:
        X = x.copy()
        fft.fft(X, axis=axis, norm=norm, out=X)
    else:
        X = fft.fft(x, axis=axis, norm=norm)

    Y = np.fft.fft(x, axis=axis, norm=norm)

    x_ = fft.ifft(X, axis=axis, norm=norm)

    if not np.allclose(X,Y):
        print '  Failed forward with ndim=%d axis=%d dims=' % (ndim, axis), dims
        passed = False

    if not np.allclose(x, x_):
        print '  Failed backward with ndim=%d axis=%d dims=' % (ndim, axis), dims
        passed = False

if passed:
    print '  Passed.'
else:
    print '  Failed.'

print 'Test speed:'



n_iter = 200
M = 1024
N = 15
n = 1024
axis = 1
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
    fft.fft(A, n=n, axis=axis, out=C)
mkl_time = time.time() - start_time
print("  mkl   --- %s seconds ---" % (mkl_time))
print("  Speed-up %.2f" % (npy_time / mkl_time))
