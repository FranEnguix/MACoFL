from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="macofl",
    version="0.1.0",  # modify it also in macofl.__version__
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Automatically install dependencies from requirements.txt
)
