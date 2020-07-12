import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygmo-ci4esa",
    version="2.15.0",
    author="Dario Izzo",
    author_email="dario.izzo@esa.int",
    description="A platform to perform parallel computations of optimisation tasks (global and local) via the asynchronous generalized island model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/esa/pygmo2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
