from setuptools import setup

import os
import re

HERE = os.path.abspath(os.path.dirname(__file__))

exc_folders = ['__pycache__', '__init__.py']
subpkgs = os.listdir(os.path.join(HERE,'segment3D'))
subpkgs = [pkg for pkg in subpkgs if pkg not in exc_folders]
print(subpkgs)

with open("requirements.txt", "r") as fp:
    install_requires = list(fp.read().splitlines())

setup(name='u-Segment3D',
	  version='0.1.0',
	  description='Generate consensus 3D segmentation from 2D segmented stacks',
	  author='Felix Y. Zhou',
	  packages=['segment3D'],
	#   package_dir={"": "unwrap3D"}, # directory containing all the packages (e.g.  src/mypkg, src/mypkg/subpkg1, ...)
	  include_package_data=True,
	  install_requires=install_requires,
)

