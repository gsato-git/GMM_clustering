from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="gmm-clustering",
    version="0.1.0",
    author="gsato-git",
    description="Gaussian Mixture Model clustering with PCA and CCA for climate data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gsato-git/GMM_clustering",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
