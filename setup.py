from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="utf-8") as history_file:
    history = history_file.read()

setup(
    name="macofl",
    version="0.1.0",  # modify it also in macofl.__version__
    keywords="macofl",
    description="Multilayered Asynchronous Consensus FL.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Automatically install dependencies from requirements.txt
    author="Francisco Enguix",
    author_email="enguixfco@gmail.com",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    url="https://github.com/FranEnguix/MACoFL",
    zip_safe=False,
)
