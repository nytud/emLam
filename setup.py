#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# I used the following resources to compile the packaging boilerplate:
# https://python-packaging.readthedocs.io/en/latest/
# https://packaging.python.org/distributing/#requirements-for-packaging-and-distributing

from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='emlam',
      version='1.0',
      description='Scripts and tools for Hungarian Language Modeling',
      long_description=readme(),
      url='https://github.com/DavidNemeskey/emLam',
      author='Dávid Márk Nemeskey',
      license='MIT',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          # This one is not in the list...
          'Topic :: Scientific/Engineering :: Natural Language Processing',

          # Environment
          'Operating System :: POSIX :: Linux',
          'Environment :: Console',
          'Natural Language :: English',

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='language modeling Hungarian RNN',
      packages=find_packages(exclude=['scripts']),
      # Include the configuration -- unfortunately, MANIFEST.in doesn't seem
      # to do it for bdist (and package_data for sdist)
      package_data={
          'conf': ['*'],
      },
      # Install the scripts
      scripts=[
          'scripts/extract_embedding.py',
          'scripts/lstm_lm.py',
          'scripts/preprocess_corpus.py',
          'scripts/shuffle_in_memory.py',
          'scripts/tsv_to_running_text.py',
          'scripts/uniq_corpus.py',
          'scripts/unk_corpus.py'
      ],
      # Tensorflow and numpy can be installed from requirement files, as they
      # are only required if the nn module / scripts are used.
      install_requires=[
          # Python 2/3 compatibility
          'future', 'six',
          # Communication with GATE; an earlier lxml version is broken, but
          'requests',
          # An earlier version was broken (don't remember which); also,
          # iterparse is broken in 3.7.3. :)
          'lxml==3.6.4',
          # Better configuration files.
          'configobj'
      ],
      # zip_safe=False,
      use_2to3=False)
