''' 
Wrapper for the MKL FFT routines.

Inspiration from:
http://stackoverflow.com/questions/11752898/threaded-fft-in-enthought-python
'''

import numpy as np
import ctypes as _ctypes
import os

from dftidefs import *

if os.name == 'posix':
    mkl = _ctypes.cdll.LoadLibrary("libmkl_rt.dylib")
else:
    try:
        mkl = _ctypes.cdll.LoadLibrary("mk2_rt.dll")
    except:
        raise ValueError('MKL Library not found')


def mkl_rfft(a, n=None, axis=-1, norm=None, direction='forward', out=None):
    ''' 
    Forward one-dimensional double-precision real-complex FFT.
    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # Add zero padding if needed (incurs memory copy)
    if n is not None:
        pad_width = np.zeros((a.ndim, 2))
        pad_width[axis,1] = n - a.shape[axis]
        a = np.pad(x, pad_width, mode='constant')
    else:
        n = a.shape[axis]

    order = 'C'
    if a.flags['F_CONTIGUOUS']:
        order = 'F'

    out_type = np.complex128
    if a.dtype == np.float32:
        out_type = np.complex64

    # Configure in-place vs out-of-place
    if out is not None:
        assert out.dtype == out_type
        for i in xrange(a.ndim):
            if i != axis:
                assert a.shape[i] == out.shape[i]
        assert (n+1)/2 == out.shape
        assert not np.may_share_memory(a, out)
    else:
        size = list(a.shape)
        size[axis] = n//2 + 1
        out = np.empty(size, dtype=out_type, order=order)

    # Define length, number of transforms strides
    length = _ctypes.c_int(n)
    n_transforms = _ctypes.c_int(np.prod(a.shape)/a.shape[axis])

    # For strides, the C type used *must* be int64
    strides = (_ctypes.c_int64*2)(0, a.strides[axis]/a.itemsize)
    if a.flags['C_CONTIGUOUS']:
        if a.ndim != 1 and (axis == -1 or axis == a.ndim-1):
            distance = _ctypes.c_int(a.shape[axis])
            out_distance = _ctypes.c_int(out.shape[axis])
        else:
            distance = _ctypes.c_int(1)
            out_distance = _ctypes.c_int(1)
    elif a.flags['F_CONTIGUOUS']:
        if a.ndim != 1 and axis == 0:
            distance = _ctypes.c_int(a.shape[axis])
            out_distance = _ctypes.c_int(out.shape[axis])
        else:
            distance = _ctypes.c_int(1)
            out_distance = _ctypes.c_int(1)
    else:
        assert False

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if a.dtype == np.float32:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_REAL, _ctypes.c_int(1), length)
    elif a.dtype == np.float64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_REAL, _ctypes.c_int(1), length)

    # set the storage type
    mkl.DftiSetValue(Desc_Handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)

    # set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1/np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1/np.sqrt(a.shape[axis]))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1./a.shape[axis])
        else:
            scale = _ctypes.c_double(1./a.shape[axis])
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, out_distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    # Not-in-place FFT
    mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

    mkl.DftiCommitDescriptor(Desc_Handle)
    fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )


    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out



def mkl_fft(a, n=None, axis=-1, norm=None, direction='forward', out=None):
    ''' 
    Forward/Backward one-dimensional single/double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # Add zero padding if needed (incurs memory copy)
    if n is not None:
        pad_width = np.zeros((a.ndim, 2))
        pad_width[axis,1] = n - a.shape[axis]
        a = np.pad(x, pad_width, mode='constant')

    # Convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Define length, number of transforms strides
    length = _ctypes.c_int(a.shape[axis])
    n_transforms = _ctypes.c_int(np.prod(a.shape)/a.shape[axis])
    
    # For strides, the C type used *must* be int64
    strides = (_ctypes.c_int64*2)(0, a.strides[axis]/a.itemsize)
    if a.flags['C_CONTIGUOUS']:
        if a.ndim != 1 and (axis == -1 or axis == a.ndim-1):
            distance = _ctypes.c_int(a.shape[axis])
        else:
            distance = _ctypes.c_int(1)
    elif a.flags['F_CONTIGUOUS']:
        if a.ndim != 1 and axis == 0:
            distance = _ctypes.c_int(a.shape[axis])
        else:
            distance = _ctypes.c_int(1)
    else:
        assert False

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(1), length)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(1), length)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1/np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1/np.sqrt(a.shape[axis]))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1./a.shape[axis])
        else:
            scale = _ctypes.c_double(1./a.shape[axis])
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )


    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def fft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='forward', out=out)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='backward', out=out)


def mkl_fft2(a, norm=None, direction='forward', out=None):
    ''' 
    Forward two-dimensional double-precision complex-complex FFT.
    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int64*2)(*a.shape)
   
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)


    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1.0/np.sqrt(np.prod(a.shape)))
        else:
            scale = _ctypes.c_double(1.0/np.sqrt(np.prod(a.shape)))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1.0/np.prod(a.shape))
        else:
            scale = _ctypes.c_double(1.0/np.prod(a.shape))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)


    # Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_int*3)(0, a.strides[0]/a.itemsize, a.strides[1]/a.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )


    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        # Set output strides if necessary
        if not out.flags['C_CONTIGUOUS']:
            out_strides = (_ctypes.c_int*3)(0, out.strides[0]/out.itemsize, out.strides[1]/out.itemsize)
            mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def fft2(a, norm=None, out=None):
    return mkl_fft2(a, norm=norm, direction='forward', out=out)


def ifft2(a, norm=None, out=None):
    return mkl_fft2(a, norm=norm, direction='backward', out=out)


if __name__ == "__main__":

    import time

    n_iter = 200
    N = 256

    np.seterr(all='raise')

    start_time = time.time()
    C = np.zeros((N, N), dtype='complex128')
    for i in range(n_iter):
        A = np.random.randn(N, N)
        C += np.fft.fft2(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    C = np.zeros((N, N), dtype='complex64')
    for i in range(n_iter):
        A = np.complex64(np.random.randn(N, N))
        C += fft2(A)
    print("--- %s seconds ---" % (time.time() - start_time))


