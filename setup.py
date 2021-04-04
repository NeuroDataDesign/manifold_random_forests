from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

# Cythonize splitter
ext_modules = [
    Extension(
        "split",
        ["oblique_forests/split.pyx"],
        extra_compile_args=[
            "-O3",
        ],
        extra_link_args=[
            "-O3"
        ],
        language="c++",
    )
]

with open("requirements.txt", mode="r", encoding="utf8") as f:
    REQUIREMENTS=f.read()

setup(
    name="oblique_forests",
    version=0.1,
    author="Adam Li, Chester Huynh, Parth Vora",
    # skipping author email, maintainer, maintainer email,
    description="A package to implement and extend SPORF and MORF",
    license="MIT",
    install_requirements=REQUIREMENTS,
    packages=find_packages(exclude=["oblique_forests/tests/*"]),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
)
