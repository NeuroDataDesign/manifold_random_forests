import os

from morf._build_utils import maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('morf', parent_package, top_path)

    # submodules which have their own setup.py
    config.add_subpackage('tree')

    # add the test directory
    # config.add_subpackage('tests')

    print('inside here....', top_path)
    print(config)
    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
