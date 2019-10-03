import setuptools, os

PACKAGE_NAME = 'xt-training'
VERSION = '0.1.0'
AUTHOR = 'Xtract Technologies'
EMAIL = 'info@xtract.ai'
DESCRIPTION = 'Utilities for training models in pytorch'
GITHUB_URL = 'https://github.com/XtractTech/xt-training'

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f'{parent_dir}/README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    packages=[
        'xt_training'
    ],
    provides=['xt_training'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn'
    ],
)
