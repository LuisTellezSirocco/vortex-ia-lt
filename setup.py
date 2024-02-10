from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

# get version
with open("src/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, version = line.replace("'", "").split()
            version = version.replace('"', "")

setup(
    name="vortex-ia-lt",
    version=version,
    license="MIT",
    description="""Framework for the development of machine learning models for vortex competition.""",
    author="Luis Téllez Ramírez",
    author_email="luistellez@sirocco.energy",
    company="Sirocco Energy",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
