from setuptools import setup, find_packages

setup(
    name="rbsp",
    version="0.1.1",
    description="Custom image file format (RBSP) library.",
    author="Alex Chica Vaquerizo",
    author_email="axcavo@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy~=2.5.3",
        "pillow~=12.3.0",
    ],
    license="MIT",
)