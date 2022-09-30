from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='locator',
      version='1.1',
      description='supervised machine learning of geographic location from genetic variation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kern-lab/locator',
      license='NPOSL-3.0',
      packages=find_packages(exclude=[]),
      install_requires=["numpy",
                        "tensorflow",
                        "h5py",
                        "scikit-allel",
                        "matplotlib",
                        "scipy",
                        "tqdm",
                        "pandas",
                        "zarr",
                        "seaborn"],
      scripts=["scripts/locator.py"],
      zip_safe=False,
      setup_requires=["numpy"]
)
