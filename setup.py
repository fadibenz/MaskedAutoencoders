from setuptools import setup, find_packages

setup(
    name='MaskedAutoencoder',
    version='0.1.0',
    description='',
    author='Fadi Benzaima',
    packages=find_packages(where='MaskedAutoencoder'),
    package_dir={'': 'MaskedAutoencoder'},
    install_requires=[
        'torch',
        'torchvision',
        'einops'
    ],
)
