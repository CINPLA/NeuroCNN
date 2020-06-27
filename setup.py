from setuptools import setup, find_packages

d = {}
exec(open("neurocnn/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "neurocnn"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Paolo Buccino, Michael Kordovan",
    author_email="alessiop.buccino@gmail.com",
    description="Python package for localization and classification of neurons from extracellular spikes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CINPLA/NeuroCNN",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'MEArec',
        'tensorflow'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
