#!/usr/bin/env python3

from setuptools import setup

version = '0.0.0'
author = 'Joha Park'
description = '''
    polya: Codes to reproduce some key results from the LARP1-poly(A) study.
'''

requirements = list(map(str.strip, open('requirements.txt').readlines()))

setup(
    name="polya",
    version=version,
    author=author,
    description=description,
    install_requires=requirements,
)
