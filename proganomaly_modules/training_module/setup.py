from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name="proganomaly_module",
    version="0.1",
    author="",
    author_email="",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Proganomaly TF2 model.",
    requires=[]
)