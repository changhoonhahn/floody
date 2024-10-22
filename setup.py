from setuptools import setup, find_packages


long_description = open('README.md').read()

setup(
    name="floody",
    description="forecasting flood losses and savings",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.0.0",
    license="MIT",
    author="ChangHoon Hahn",
    author_email="changhoon.hahn@princeton.edu",
    url="https://github.com/changhoonhahn/floody",
    packages=find_packages(where="src"), 
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
    ],
    keywords = ['machinelearning'],
    install_requires=["torch", "numpy", "astropy", "sbi", "geopandas"]
)
