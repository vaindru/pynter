#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:39:37 2023

@author: villa
"""

from setuptools import setup, find_namespace_packages

setup(
    name='pynter',
    version='1.0.1',
    author='Lorenzo Villa',
    description='Tools for atomistic calculations',
    packages=find_namespace_packages(exclude=["pynter.tests","pynter.*.tests", "pynter.*.*.tests"]),
    install_requires=[
        'ase',
        'pymatgen>=2023.3.23',
        'pymatgen-analysis-defects>=2023.3.25',
        'pymatgen-db',
        'PyYAML',
        'schedule'        
    ],
    extra_requires={'test':'pytest'},
    entry_points={
    "console_scripts": [
        "pynter = pynter.cli.main:main"]}
)

