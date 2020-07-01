from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='locator',
      version='1.1',
      description='supervised machine learning of geographic location from genetic variation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kern-lab/locator',
      author='CJ Battey',
      author_email='cjbattey@gmail.com',
      license='NPOSL-3.0',
      packages=find_packages(exclude=[]),
      install_requires=["tensorflow",
                        "numpy",
                        "h5py",
                        "scikit-allel",
                        "matplotlib",
                        "scipy==1.4.1",
                        "tqdm",
                        "pandas",
                        "zarr",
                        "gnuplotlib"],
      scripts=["scripts/locator.py"],
      zip_safe=False,
      setup_requires=[]
)
