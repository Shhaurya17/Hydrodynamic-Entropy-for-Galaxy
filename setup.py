from setuptools import setup, find_packages

setup(
    name="hydro_spectral",
    version="1.0.0",
    description="Hydrodynamic spectral entropy, FFT power spectra, Helmholtz decomposition, and power-law fitting for cosmological simulations",
    author="Physics Analysis Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas"
    ],
    python_requires=">=3.8",
)
