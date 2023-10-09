#!/usr/bin/env python

import setuptools
import yandex_chain

with open('README.md') as readme_file:
    readme = readme_file.read()
    
setuptools.setup(
    name='yandex-chain',
    packages=setuptools.find_packages('yandex_chain',exclude=('tests',)),
    version=yandex_chain.__version__,
    install_requires=['requests','langchain','tenacity'],
    description='Yandex GPT Support for LangChain',
    author='Dmitri Soshnikov',
    author_email='dmitri@soshnikov.com',
    url='https://github.com/yandex-datasphere/yandex-chain',
    long_description=readme,
    long_description_content_type='text/markdown; charset=UTF-8',
    license='MIT license',
    classifiers=[
        "Programming Language :: Python :: 3",
#        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    test_suite = 'tests'
)