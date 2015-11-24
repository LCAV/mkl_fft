''' Wrapper to MKL FFT routines '''

import sys
import numpy as np
import ctypes as _ctypes

import os

if os.name == 'posix':
    mkl = _ctypes.cdll.LoadLibrary("/Users/scheibler/anaconda//lib/libmkl_rt.dylib")
else:
    try:
        mkl = _ctypes.cdll.LoadLibrary("mk2_rt.dll")
    except:
        raise ValueError('MKL Library not found')

_DFTI_COMPLEX = _ctypes.c_int(32)
_DFTI_DOUBLE = _ctypes.c_int(36)
_DFTI_NOT_INPLACE = _ctypes.c_int(44)

_DFTI_PLACEMENT = _ctypes.c_int(11)
_DFTI_INPUT_STRIDES = _ctypes.c_int(12)
_DFTI_OUTPUT_STRIDES = _ctypes.c_int(13)

# enum DFTI_CONFIG_PARAM from mkl_dfti.h
_DFTI_FORWARD_DOMAIN = _ctypes.c_int(0)  # Domain for forward transform, no default */
_DFTI_DIMENSION      = _ctypes.c_int(1)  # Dimension, no default */
_DFTI_LENGTHS        = _ctypes.c_int(2)  # length(s) of transform, no default */
_DFTI_PRECISION      = _ctypes.c_int(3)  # Precision of computation, no default */
_DFTI_FORWARD_SCALE  = _ctypes.c_int(4)  # Scale factor for forward transform, default = 1.0 */
_DFTI_BACKWARD_SCALE = _ctypes.c_int(5)  # Scale factor for backward transform, default = 1.0 */
_DFTI_FORWARD_SIGN   = _ctypes.c_int(6)  # Default for forward transform = DFTI_NEGATIVE  */
_DFTI_NUMBER_OF_TRANSFORMS = _ctypes.c_int(7)   # Number of data sets to be transformed, default = 1 */
_DFTI_COMPLEX_STORAGE = _ctypes.c_int(8) # Representation for complex domain, default = DFTI_COMPLEX_COMPLEX */
_DFTI_REAL_STORAGE           = _ctypes.c_int(9) # Rep. for real domain, default = DFTI_REAL_REAL */
_DFTI_CONJUGATE_EVEN_STORAGE = _ctypes.c_int(10) # Rep. for conjugate even domain, default = DFTI_COMPLEX_REAL */
_DFTI_PLACEMENT      = _ctypes.c_int(11)    # Placement of result, default = DFTI_INPLACE */
_DFTI_INPUT_STRIDES  = _ctypes.c_int(12)    # Stride information of input data, default = tigthly */
_DFTI_OUTPUT_STRIDES = _ctypes.c_int(13)    # Stride information of output data, default = tigthly */
_DFTI_INPUT_DISTANCE  = _ctypes.c_int(14)   # Distance information of input data, default = 0 */
_DFTI_OUTPUT_DISTANCE = _ctypes.c_int(15)   # Distance information of output data, default = 0 */
_DFTI_INITIALIZATION_EFFORT = _ctypes.c_int(16) # Effort spent in initialization, default = DFTI_MEDIUM */
_DFTI_WORKSPACE   = _ctypes.c_int(17)    # Use of workspace during computation, default = DFTI_ALLOW */
_DFTI_ORDERING    = _ctypes.c_int(18)    # Possible out of order computation, default = DFTI_ORDERED */
_DFTI_TRANSPOSE   = _ctypes.c_int(19)    # Possible transposition of result, default = DFTI_NONE */
_DFTI_DESCRIPTOR_NAME = _ctypes.c_int(20)  # name of descriptor, default = string of zero length */
_DFTI_PACKED_FORMAT = _ctypes.c_int(21)    # packed format for real transform, default = DFTI_CCS_FORMAT */
# below 4 parameters for get_value functions only */
_DFTI_COMMIT_STATUS     = _ctypes.c_int(22)     # Whether descriptor has been commited */
_DFTI_VERSION           = _ctypes.c_int(23)     # DFTI implementation version number */
_DFTI_FORWARD_ORDERING  = _ctypes.c_int(24)     # The ordering of forward transform */
_DFTI_BACKWARD_ORDERING = _ctypes.c_int(25)     # The ordering of backward transform */
# below for set_value and get_value functions */
_DFTI_NUMBER_OF_USER_THREADS = _ctypes.c_int(26) # number of user's threads) default = 1 */

# DFTI options values
_DFTI_COMMITTED     = _ctypes.c_int(30)   # /* status - commit */
_DFTI_UNCOMMITTED   = _ctypes.c_int(31)   # /* status - uncommit */
_DFTI_COMPLEX      = _ctypes.c_int(32)    #/* General domain */
_DFTI_REAL         = _ctypes.c_int(33)    #/* Real domain */
# _DFTI_CONJUGATE_EVEN = _ctypes.c_int(34)  #/* Conjugate even domain */
_DFTI_SINGLE       = _ctypes.c_int(35)    #/* Single precision */
_DFTI_DOUBLE       = _ctypes.c_int(36)    #/* Double precision */
# _DFTI_NEGATIVE     = _ctypes.c_int(37)    #/* -i, for setting definition of transform */
# _DFTI_POSITIVE     = _ctypes.c_int(38)    #/* +i, for setting definition of transform */
_DFTI_COMPLEX_COMPLEX = _ctypes.c_int(39) #/* Representation method for domain */
_DFTI_COMPLEX_REAL   = _ctypes.c_int(40) #/* Representation method for domain */
_DFTI_REAL_COMPLEX   = _ctypes.c_int(41) #/* Representation method for domain */
_DFTI_REAL_REAL    = _ctypes.c_int(42)   #/* Representation method for domain */
_DFTI_INPLACE      = _ctypes.c_int(43)   #/* Result overwrites input */
_DFTI_NOT_INPLACE  = _ctypes.c_int(44)   #/* Result placed differently than input */
# _DFTI_LOW          = _ctypes.c_int(45)   #/* A low setting */
# _DFTI_MEDIUM       = _ctypes.c_int(46)   #/* A medium setting */
# _DFTI_HIGH         = _ctypes.c_int(47)   #/* A high setting */
_DFTI_ORDERED      = _ctypes.c_int(48)   #/* Data on forward and backward domain ordered */
_DFTI_BACKWARD_SCRAMBLED  = _ctypes.c_int(49) #  /* Data on forward ordered and backward domain scrambled */
# _DFTI_FORWARD_SCRAMBLED   = _ctypes.c_int(50)#   /* Data on forward scrambled and backward domain ordered */
_DFTI_ALLOW        = _ctypes.c_int(51) #  /* Allow certain request or usage */
# _DFTI_AVOID        = _ctypes.c_int(52) #  /* Avoid certain request or usage */
_DFTI_NONE         = _ctypes.c_int(53) #  /* none certain request or usage */
_DFTI_CCS_FORMAT   = _ctypes.c_int(54) #/* ccs format for real DFT */
_DFTI_PACK_FORMAT  = _ctypes.c_int(55) # /* pack format for real DFT */
_DFTI_PERM_FORMAT  = _ctypes.c_int(56) #  /* perm format for real DFT */
_DFTI_CCE_FORMAT   = _ctypes.c_int(57) #   /* cce format for real DFT */

def mkl_rfft(a, n=None, axis=-1, norm=None, direction='forward', out=None):
    ''' 
    Forward one-dimensional double-precision real-complex FFT.
    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.
    '''

    # this code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # add zero padding if needed (incurs memory copy)
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
    # for strides, the C type used *must* be int64
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
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_SINGLE, _DFTI_REAL, _ctypes.c_int(1), length)
    elif a.dtype == np.float64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_DOUBLE, _DFTI_REAL, _ctypes.c_int(1), length)

    # set the storage type
    mkl.DftiSetValue(Desc_Handle, _DFTI_CONJUGATE_EVEN_STORAGE, _DFTI_COMPLEX_COMPLEX)

    # set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1/np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1/np.sqrt(a.shape[axis]))
        mkl.DftiSetValue(Desc_Handle, _DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, _DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1./a.shape[axis])
        else:
            scale = _ctypes.c_double(1./a.shape[axis])
        mkl.DftiSetValue(Desc_Handle, _DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, _DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_DISTANCE, out_distance)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    #Not-inplace FFT
    mkl.DftiSetValue(Desc_Handle, _DFTI_PLACEMENT, _DFTI_NOT_INPLACE)

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

    # this code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # add zero padding if needed (incurs memory copy)
    if n is not None:
        pad_width = np.zeros((a.ndim, 2))
        pad_width[axis,1] = n - a.shape[axis]
        a = np.pad(x, pad_width, mode='constant')

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
        assert out.dtype == np.complex128
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Define length, number of transforms strides
    length = _ctypes.c_int(a.shape[axis])
    n_transforms = _ctypes.c_int(np.prod(a.shape)/a.shape[axis])
    # for strides, the C type used *must* be int64
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
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_SINGLE, _DFTI_COMPLEX, _ctypes.c_int(1), length)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), _DFTI_DOUBLE, _DFTI_COMPLEX, _ctypes.c_int(1), length)

    # set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1/np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1/np.sqrt(a.shape[axis]))
        mkl.DftiSetValue(Desc_Handle, _DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, _DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1./a.shape[axis])
        else:
            scale = _ctypes.c_double(1./a.shape[axis])
        mkl.DftiSetValue(Desc_Handle, _DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, _DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, _DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, _DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        #Inplace FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        #Not-inplace FFT
        mkl.DftiSetValue(Desc_Handle, _DFTI_PLACEMENT, _DFTI_NOT_INPLACE)

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )


    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out

def fft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='forward', out=out)

def ifft(a, n=None, axis=-1, norm=None, out=None):
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='backward', out=out)


if __name__ == "__main__":


    max_dim = 2
    min_len = 1
    max_len = 100

    ndim = np.random.randint(1,high=max_dim+1)
    axis = np.random.randint(ndim)

    print 'Test mkl fft'
    print 'ndim',ndim
    print 'axis=',axis

    dims = np.random.randint(min_len, high=max_len, size=(ndim))

    print 'Test complex FFT'
    x = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)

    X = fft(x, axis=axis)
    Y = np.fft.fft(x, axis=axis)
    print 'forward', np.allclose(X, Y)

    x_ = ifft(X, axis=axis)
    print 'backward', np.allclose(x, x_)

    xr = np.random.normal(size=dims)

    Xr = mkl_rfft(xr, axis=axis)
    Yr = np.fft.rfft(xr, axis=axis)
    print 'forward', np.allclose(Xr, Yr)

    #xr_ = ifft(Xr, axis=axis)
    #print 'backward', np.allclose(Xr, Yr)



