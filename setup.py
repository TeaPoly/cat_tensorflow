#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'librosa',
    'numpy',
    'tqdm',
    'python_speech_features'
]

setup_requirements = []

test_requirements = []

setup(
    author="huanglk",
    author_email='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Crf-based ASR toolkit based on TensorFlow. (https://github.com/TeaPoly)",
    entry_points={
        'console_scripts': [
            'cat_tensorflow=cat_tensorflow.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description="",  # readme + '\n\n' + history,
    include_package_data=True,
    keywords='cat_tensorflow',
    name='cat_tensorflow',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='',
    version='0.1.0',
    zip_safe=False,
)
