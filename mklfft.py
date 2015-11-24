''' 
Wrapper for the MKL FFT routines.

Inspiration from:
http://stackoverflow.com/questions/11752898/threaded-fft-in-enthought-python
'''

import numpy as _np
import ctypes as _ctypes

mkl = _ctypes.cdll.LoadLibrary("libmkl_rt.dylib")

_DFTI_REAL           = _ctypes.c_int(33)
_DFTI_COMPLEX        = _ctypes.c_int(32)
_DFTI_SINGLE         = _ctypes.c_int(35)
_DFTI_DOUBLE         = _ctypes.c_int(36)
_DFTI_PLACEMENT      = _ctypes.c_int(11)
_DFTI_NOT_INPLACE    = _ctypes.c_int(44)
_DFTI_INPUT_STRIDES  = _ctypes.c_int(12)
_DFTI_OUTPUT_STRIDES = _ctypes.c_int(13)

def fft2(a, out=None):
    ''' 
    Forward two-dimensional double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    assert a.dtype == _np.complex128
    assert len(a.shape) == 2

    inplace = False

    if out is a:
        inplace = True

    elif out is not None:
        assert out.dtype == _np.complex128
        assert a.shape == out.shape
        assert not _np.may_share_memory(a, out)

    else:
        out = _np.empty_like(a)

    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int*2)(*a.shape)

    mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_DOUBLE, _DFTI_COMPLEX, _ctypes.c_int(2), dims )

    #Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_int*3)(0, a.strides[0]/16, a.strides[1]/16)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if inplace:
        #Inplace FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        mkl.DftiComputeForward(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        #Not-inplace FFT
        mkl.DftiSetValue(Desc_Handle, _DFTI_PLACEMENT, _DFTI_NOT_INPLACE)

        #Set output strides if necessary
        if not out.flags['C_CONTIGUOUS']:
            out_strides = (_ctypes.c_int*3)(0, out.strides[0]/16, out.strides[1]/16)
            mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

        mkl.DftiCommitDescriptor(Desc_Handle)
        mkl.DftiComputeForward(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out

def ifft2(a, out=None):
    ''' 
    Backward two-dimensional double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    assert a.dtype == _np.complex128
    assert len(a.shape) == 2

    inplace = False

    if out is a:
        inplace = True

    elif out is not None:
        assert out.dtype == _np.complex128
        assert a.shape == out.shape
        assert not _np.may_share_memory(a, out)

    else:
        out = _np.empty_like(a)

    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int*2)(*a.shape)

    mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_DOUBLE, _DFTI_COMPLEX, _ctypes.c_int(2), dims )

    #Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_int*3)(0, a.strides[0]/16, a.strides[1]/16)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if inplace:
        #Inplace FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        mkl.DftiComputeBackward(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        #Not-inplace FFT
        mkl.DftiSetValue(Desc_Handle, _DFTI_PLACEMENT, _DFTI_NOT_INPLACE)

        #Set output strides if necessary
        if not out.flags['C_CONTIGUOUS']:
            out_strides = (_ctypes.c_int*3)(0, out.strides[0]/16, out.strides[1]/16)
            mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

        mkl.DftiCommitDescriptor(Desc_Handle)
        mkl.DftiComputeBackward(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out



if __name__ is "__main__":
    n_iter = 100
    N = 512

    start_time = time.time()
    C = _np.zeros((N, N), dtype='complex128')
    for i in range(n_iter):
        A = _np.random.randn(N, N)
        C += _np.fft.fft2(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    C = _np.zeros((N, N), dtype='complex128')
    for i in range(n_iter):
        A = _np.complex128(_np.random.randn(N, N))
        C += fft2(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    C = _np.zeros((N, N), dtype='complex128')
    for i in range(n_iter):
        A = _np.complex128(_np.random.randn(N, N))
        C += _fft2(A)
    print("--- %s seconds ---" % (time.time() - start_time))
