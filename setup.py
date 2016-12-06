#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='mkl_fft',
      version='1.0',
      description='A wrapper for Intel MKL FFT library.',
      author='Robin Scheibler, Ivan Dokmanic',
      author_email='robin.scheibler@epfl.ch',
      url='http://lcav.epfl.ch',
      packages=['mkl_fft'],
	  install_requires=[
	      'numpy',
		  'scipy'],
      test_suite='nose.collector',
      tests_require=['nose']
)
