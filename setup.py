from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="locator",
    version="1.2.1",
    description="supervised machine learning of geographic location from genetic variation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kr-colab/locator",
    license="NPOSL-3.0",
    packages=find_packages(exclude=[]),
    install_requires=[
        "numpy>=1.20.0,<1.25.0",
        "tensorflow>=2.10.0",
        "h5py",
        "scikit-allel",
        "matplotlib",
        "scipy",
        "tqdm",
        "pandas",
        "zarr",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "locator=locator.locator:main",
        ],
    },
    zip_safe=False,
    python_requires=">=3.8,<3.12",
    setup_requires=["numpy>=1.20.0"],
    author="Andrew Kern",
    author_email="adkern@uoregon.edu",
    project_urls={
        "Bug Reports": "https://github.com/kr-colab/locator/issues",
        "Documentation": "https://github.com/kr-colab/locator#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
