import setuptools

setuptools.setup(
    name="SpinDry",
    version="0.0.8",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Contains MC algorithm for generating host-guest conformers.",
    url="https://github.com/andrewtarzia/SpinDry",
    packages=setuptools.find_packages(),
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
