from setuptools import setup
from setuptools import find_packages

setup(name='dgvae',
      version='0.0.1',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'networkx',
                        'scikit-learn',
                        'scipy',
                        ],
      extras_require={
          'visualization': ['matplotlib'],
      },
      package_data={'dgvae': ['README.md']},
      packages=find_packages())
