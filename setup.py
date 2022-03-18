from setuptools import find_packages, setup
import re
from os.path import join


def get_version():
    with open(join('src', 'stk', '__init__.py'), 'r') as f:
        content = f.read()
    p = re.compile(r'^__version__ = [\'"]([^\'\"]*)[\'"]', re.M)
    return p.search(content).group(1)


setup(
    name="SpinDry",
    version="0.0.91",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Contains MC algorithm for generating host-guest conformers.",
    url="https://github.com/andrewtarzia/SpinDry",
    packages=find_packages(),
    install_requires=(
        'scipy',
        'matplotlib',
        'networkx',
        'numpy',
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
