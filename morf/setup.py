import os

from sklearn._build_utils import maybe_cythonize_extensions
from sklearn._build_utils.deprecated_modules import (
    _create_deprecated_modules_files
)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    _create_deprecated_modules_files()

    config = Configuration('morf', parent_package, top_path)

    # submodules which have their own setup.py
    config.add_subpackage('tree')

    # add cython extension module for isotonic regression
    config.add_extension('_isotonic',
                         sources=['_isotonic.pyx'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         )

    # add the test directory
    config.add_subpackage('tests')

    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
