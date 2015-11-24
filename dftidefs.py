import ctypes as _ctypes


# enum DFTI_CONFIG_PARAM from mkl_dfti.h

_DFTI_FORWARD_DOMAIN         = _ctypes.c_int(0)  # Domain for forward transform, no default 
_DFTI_DIMENSION              = _ctypes.c_int(1)  # Dimension, no default 
_DFTI_LENGTHS                = _ctypes.c_int(2)  # length(s) of transform, no default 
_DFTI_PRECISION              = _ctypes.c_int(3)  # Precision of computation, no default 
_DFTI_FORWARD_SCALE          = _ctypes.c_int(4)  # Scale factor for forward transform, default = 1.0 
_DFTI_BACKWARD_SCALE         = _ctypes.c_int(5)  # Scale factor for backward transform, default = 1.0 
_DFTI_FORWARD_SIGN           = _ctypes.c_int(6)  # Default for forward transform = DFTI_NEGATIVE  
_DFTI_NUMBER_OF_TRANSFORMS   = _ctypes.c_int(7)  # Number of data sets to be transformed, default = 1 
_DFTI_COMPLEX_STORAGE        = _ctypes.c_int(8)  # Representation for complex domain, default = DFTI_COMPLEX_COMPLEX 
_DFTI_REAL_STORAGE           = _ctypes.c_int(9)  # Rep. for real domain, default = DFTI_REAL_REAL 
_DFTI_CONJUGATE_EVEN_STORAGE = _ctypes.c_int(10) # Rep. for conjugate even domain, default = DFTI_COMPLEX_REAL 
_DFTI_PLACEMENT              = _ctypes.c_int(11) # Placement of result, default = DFTI_INPLACE 
_DFTI_INPUT_STRIDES          = _ctypes.c_int(12) # Stride information of input data, default = tigthly 
_DFTI_OUTPUT_STRIDES         = _ctypes.c_int(13) # Stride information of output data, default = tigthly 
_DFTI_INPUT_DISTANCE         = _ctypes.c_int(14) # Distance information of input data, default = 0 
_DFTI_OUTPUT_DISTANCE        = _ctypes.c_int(15) # Distance information of output data, default = 0 
_DFTI_INITIALIZATION_EFFORT  = _ctypes.c_int(16) # Effort spent in initialization, default = DFTI_MEDIUM 
_DFTI_WORKSPACE              = _ctypes.c_int(17) # Use of workspace during computation, default = DFTI_ALLOW 
_DFTI_ORDERING               = _ctypes.c_int(18) # Possible out of order computation, default = DFTI_ORDERED 
_DFTI_TRANSPOSE              = _ctypes.c_int(19) # Possible transposition of result, default = DFTI_NONE 
_DFTI_DESCRIPTOR_NAME        = _ctypes.c_int(20) # name of descriptor, default = string of zero length 
_DFTI_PACKED_FORMAT          = _ctypes.c_int(21) # packed format for real transform, default = DFTI_CCS_FORMAT 

# below 4 parameters for get_value functions only 
_DFTI_COMMIT_STATUS          = _ctypes.c_int(22) # Whether descriptor has been commited 
_DFTI_VERSION                = _ctypes.c_int(23) # DFTI implementation version number 
_DFTI_FORWARD_ORDERING       = _ctypes.c_int(24) # The ordering of forward transform 
_DFTI_BACKWARD_ORDERING      = _ctypes.c_int(25) # The ordering of backward transform 

# below for set_value and get_value functions 
_DFTI_NUMBER_OF_USER_THREADS = _ctypes.c_int(26) # number of user's threads) default = 1 

# DFTI options values
_DFTI_COMMITTED              = _ctypes.c_int(30) # status - commit 
_DFTI_UNCOMMITTED            = _ctypes.c_int(31) # status - uncommit 
_DFTI_COMPLEX                = _ctypes.c_int(32) # General domain 
_DFTI_REAL                   = _ctypes.c_int(33) # Real domain 
_DFTI_CONJUGATE_EVEN         = _ctypes.c_int(34) # Conjugate even domain 
_DFTI_SINGLE                 = _ctypes.c_int(35) # Single precision 
_DFTI_DOUBLE                 = _ctypes.c_int(36) # Double precision 
_DFTI_NEGATIVE               = _ctypes.c_int(37) # -i, for setting definition of transform 
_DFTI_POSITIVE               = _ctypes.c_int(38) # +i, for setting definition of transform 
_DFTI_COMPLEX_COMPLEX        = _ctypes.c_int(39) # Representation method for domain 
_DFTI_COMPLEX_REAL           = _ctypes.c_int(40) # Representation method for domain 
_DFTI_REAL_COMPLEX           = _ctypes.c_int(41) # Representation method for domain 
_DFTI_REAL_REAL              = _ctypes.c_int(42) # Representation method for domain 
_DFTI_INPLACE                = _ctypes.c_int(43) # Result overwrites input 
_DFTI_NOT_INPLACE            = _ctypes.c_int(44) # Result placed differently than input 
_DFTI_LOW                    = _ctypes.c_int(45) # A low setting 
_DFTI_MEDIUM                 = _ctypes.c_int(46) # A medium setting 
_DFTI_HIGH                   = _ctypes.c_int(47) # A high setting 
_DFTI_ORDERED                = _ctypes.c_int(48) # Data on forward and backward domain ordered 
_DFTI_BACKWARD_SCRAMBLED     = _ctypes.c_int(49) # Data on forward ordered and backward domain scrambled 
_DFTI_FORWARD_SCRAMBLED      = _ctypes.c_int(50) # Data on forward scrambled and backward domain ordered 
_DFTI_ALLOW                  = _ctypes.c_int(51) # Allow certain request or usage 
_DFTI_AVOID                  = _ctypes.c_int(52) # Avoid certain request or usage 
_DFTI_NONE                   = _ctypes.c_int(53) # none certain request or usage 
_DFTI_CCS_FORMAT             = _ctypes.c_int(54) # ccs format for real DFT 
_DFTI_PACK_FORMAT            = _ctypes.c_int(55) # pack format for real DFT 
_DFTI_PERM_FORMAT            = _ctypes.c_int(56) # perm format for real DFT 
_DFTI_CCE_FORMAT             = _ctypes.c_int(57) #  cce format for real DFT 
