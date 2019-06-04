
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='GeneticAlgorithm',  
    version='0.0.1',
    scripts=['Makefile'] ,
    author="Calebe Elias Ribeiro Brim",
    author_email="calebebrim@gmail.com",
    description="Genetic Algorithm Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calebebrim/geneticalgorithm.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )