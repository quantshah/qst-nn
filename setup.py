#!/usr/bin/env python
from setuptools import setup, find_packages

packages = find_packages(
    where="qst_nn",
    include=['data', 'training', 'models', 'inference'],
    exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
)

package_dir = {"data":"qst_nn/data",
               "training":"qst_nn/training",
               "models":"qst_nn/models",
               "inference":"qst_nn/inference"}


setup(name='qst_nn',
      version='0.0.1',
      description='Quantum state learning with deep neural networks',
      author='Shahnawaz Ahmed',
      author_email='shahnawaz.ahmed95@gmail.com',
	  packages=packages,
      package_dir=package_dir,
     )
