""" Python Project Template

https://github.com/username/python-project-template

Python Project Template is a starting point for creating a Python package. It provides all the necessary functionality for setting up a project structure, managing dependencies, and writing tests.

For more detailed information, please check the accompanying README.md.
"""
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

# get version
with open("src/package_name/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, version = line.replace("'", "").split()
            version = version.replace('"', "")

setup(
    name="template",
    version=version,
    license="MIT",
    description="""This template is a starting point for creating a Python package.""",
    author="Luis Téllez Ramírez",
    author_email="luistellez@sirocco.energy",
    company="Sirocco Energy",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
