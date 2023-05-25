from setuptools import setup, find_packages

setup(name='HD_CTBET',
      version='1.1',
      description=('Tool for brain extraction from CT images. '
                   'Fork of HD-BET from Isensee et al.'),
      url='https://github.com/CAAI/HD-CTBET',
      python_requires='>=3.8',
      author='Claes Ladefoged',
      author_email='claes.noehr.ladefoged@regionh.dk',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=[
          'numpy',
          'torch>=0.4.1',
          'scikit-image',
          'SimpleITK',
          'nnunet>=1.6.6'
      ],
      scripts=['HD_CTBET/hd-ctbet'],
      packages=find_packages(include=['HD_CTBET']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
