from setuptools import setup, find_packages

setup(
    name="entrap",
    version="1.0.0",
    author="Muhammad Akmal Husain",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "numba>=0.56.0",
        "hdbscan>=0.8.27",
        "ripser>=0.6.0",
        "kneed>=0.8.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0",
    ],
)
