from setuptools import setup, find_packages

setup(
    name="deltatfidf",
    version=0.1,
    packages=find_packages(),
    description="DeltaTfidf",
    license="MIT",
    install_requires=["numpy>=1.19", "scipy>=1.5", "scikit-learn>=0.24"],
)