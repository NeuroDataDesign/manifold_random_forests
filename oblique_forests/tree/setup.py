import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):  # noqa
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension("_oblique_tree",
                    sources=["_oblique_tree.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
                    language="c++",
    )    
    config.add_extension("_oblique_splitter",
                    sources=["_oblique_splitter.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
                    language="c++",
    )
    config.add_extension("_tree",
                    sources=["_tree.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
                    language="c++",
    )
    config.add_extension("_splitter",
                    sources=["_splitter.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
                    language="c++",
    )
    config.add_extension("_criterion",
                    sources=["_criterion.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
    )
    config.add_extension("_utils",
                    sources=["_utils.pyx"],
                    include_dirs=[numpy.get_include()],
                    libraries=libraries,
                    extra_compile_args=["-O3"],
    )

    config.add_subpackage("tests")
    config.add_data_files("_splitter.pxd")
    config.add_data_files("_tree.pxd")

    config.add_data_files("_criterion.pxd")
    config.add_data_files("_utils.pxd")
    config.add_data_files("_oblique_splitter.pxd")
    config.add_data_files("_oblique_tree.pxd")
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
