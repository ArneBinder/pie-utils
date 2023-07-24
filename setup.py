#!/usr/bin/env python

from setuptools import find_packages, setup

REQUIRED_PKGS = [
    "pytorch-ie>=0.16.0,<1.0.0",
    "asciidag",  # for visualization
]

TESTS_REQUIRE = [
    "pytest",
    "pytest-cov",
    "plotext",
]

QUALITY_REQUIRE = [
    "pre-commit",
]


EXTRAS_REQUIRE = {
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
}

setup(
    name="pie-utils",
    version="0.3.0",
    description="A collection of utilities to easy the usage of PyTorch-IE",
    author="Arne Binder",
    author_email="arne.b.binder@gmail.com",
    url="https://github.com/ArneBinder/pie-utils",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
)
