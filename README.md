# mklfft

This is a wrapper around the Intel Math Kernel Library FFT. It is very fast.

### Authors

* Ivan Dokmanic
* Robin Scheibler

### Setup the library

It might be necessary to setup an environment variable with location of
the shared library

    export LIBMKL=$HOME/anaconda/lib/libmkl_rt.dylib

for example with anaconda installed in the home directory. This should be adapted to your setup.

### Use the library

The library has a syntax similar to that of numpy.

It supports

* In-place and out-of-place transforms
* Scrambled or unscrambled output 
* Orthogonal or standard normalization
* 1D FFT, including multiple FFTs on arbitrary dimension of a multidimensional array
* 2D FFT
* Real or complex data

### License

    Copyright (c) 2016 Ivan Dokmanic, Robin Scheibler

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
