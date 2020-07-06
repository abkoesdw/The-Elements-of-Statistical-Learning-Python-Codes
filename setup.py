from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    long_description = fh.read()

setup(
    name="esl",
    version="0.0.1",
    author="Arief Koesdwiady",
    author_email="ariefbarkah@gmail.com",
    description="This repo contains python implementation of the book Elements of Statistical Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abkoesdw/The-Elements-of-Statistical-Learning-Python-Codes.git",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.2.2",
        "scipy>=1.5.1",
        "tqdm>=4.47.0",
        "flake8>=3.8.3",
        "black>=19.10b0",
        "ml-datasets@git+https://github.com/abkoesdw/ml-datasets.git@dev",
        # "jupyterlab>=2.1.5",
    ],
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)
