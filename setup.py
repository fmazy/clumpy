import setuptools
from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
setup(
      ext_modules=cythonize("clumpy/metrics/_mcdf.pyx")
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clumpy",
    version="0.1.0",
    author="François-Rémi Mazy",
    author_email="francois-remi.mazy@inria.fr",
    description="Comprehensive Land Use Models in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/fmazy/clumpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
