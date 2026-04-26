"""
setup.py -- compatibility shim for tools that do not support PEP 517/518.
Primary configuration lives in pyproject.toml.

Compatible with Python 3.8 through 3.14.
"""

from __future__ import annotations

import sys
import os
import re

_MIN = (3, 8)
_MAX = (3, 15)

if sys.version_info < _MIN:
    sys.exit(
        "ERROR: This project requires Python {}.{}+.\n"
        "You are running Python {}.{}.{}".format(
            _MIN[0], _MIN[1], *sys.version_info[:3]
        )
    )

if sys.version_info >= _MAX:
    import warnings
    warnings.warn(
        "Python {}.{}.{} has not been tested. Proceed with caution.".format(
            *sys.version_info[:3]
        ),
        RuntimeWarning,
        stacklevel=1,
    )

try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("setuptools is not installed. Run:  pip install setuptools")

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(filename):
    path = os.path.join(HERE, filename)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _parse_requirements(filename):
    """Parse a requirements file into a list of dependency strings."""
    reqs = []
    for line in _read(filename).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "-c", "-i", "--")):
            continue
        line = re.split(r"\s+#", line)[0].strip()
        if line:
            reqs.append(line)
    return reqs


def _py_classifiers():
    versions = [(3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14)]
    return [
        "Programming Language :: Python :: {}.{}".format(maj, min_)
        for maj, min_ in versions
    ]


INSTALL_REQUIRES = _parse_requirements("requirements.txt")
EXTRAS_WEB = _parse_requirements("requirements-web.txt")

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
] + _py_classifiers() + [
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
]

setup(
    name="duck-egg-fertility-detection",
    version="1.0.0",
    description="Duck Egg Fertility Detection using Deep Learning and Clustering",
    long_description=_read("readme.md"),
    long_description_content_type="text/markdown",
    author="Yumiko Pubangelo",
    author_email="wkartiwa35@gmail.com",
    url="https://github.com/yumikopubangelo/duck_egg_fertility_detection",
    license="MIT",
    packages=find_packages(
        where=".",
        include=["src*", "app*", "web*", "scripts*"],
        exclude=["tests*", "docs*", "deployment*", "notebooks*", "data*"],
    ),
    package_dir={"": "."},
    python_requires=">=3.8, <4",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "web": EXTRAS_WEB,
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "all": EXTRAS_WEB + [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "duck-train   = scripts.03_train_unet:main",
            "duck-predict = scripts.07_inference:main",
            "duck-serve   = web.api.app:main",
        ],
    },
    classifiers=CLASSIFIERS,
    keywords=[
        "deep learning", "image classification", "duck egg",
        "fertility detection", "computer vision", "unet", "pytorch",
    ],
    include_package_data=True,
    zip_safe=False,
)
