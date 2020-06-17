from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['mphpo'],
    package_dir={'': 'src'},
    install_requires=['hpbandster']
)

setup(**setup_args)
