from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize



import numpy as np
import os,sys,subprocess,cython


os.environ['ARCHFLAGS'] = "-arch x86_64"

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"

this_dir = os.path.split(cython.__file__)[0]

setup(
  name = 'options',
  ext_modules = cythonize(["*.pyx","time/kit_time.pyx"]),
  include_dirs=[np.get_include(),this_dir],
  language="c++"
)
