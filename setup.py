import setuptools, os

PACKAGE_NAME = 'xt-training'
VERSION = '1.11.0'
AUTHOR = 'Xtract AI'
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
        'xt_training',
        'xt_training.utils',
        'xt_training.utils.nni'
    ],
    package_data={'': ['*.json', '*.yml']},
    provides=['xt_training'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
        'plotly',
        'pynvml',
        'nni'
    ],
)
