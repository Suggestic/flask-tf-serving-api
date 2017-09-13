#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name='Flask-TFServing',
    version='0.1',
    description='TF serving support for Flask applications',
    #long_description='Multi Label Ontology Model',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Programming Language :: Python :: 2.7',
        'Topic :: Deep Learning :: AI',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='deep learning label ontology classification regression label',
    #download_url='http://github.com/Suggestic/mlom',
    author='Ismael Fernández',
    author_email='ismael@suggestic.com',
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    zip_safe=False)
