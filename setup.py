from setuptools import setup, find_packages

setup(
   name='volterra-point_processes',
   version='1.0',
   description='Volterra process simulation and solvers for Volterra equations',
   author='MrG1raffe',
   author_email='dimitri.sotnikov@gmail.com',
   packages=find_packages(),
   install_requires=['numpy', 'matplotlib', 'scipy'], #external packages as dependencies
)