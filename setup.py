from setuptools import setup, find_packages

setup(
    name="cifar10_custom",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'albumentations>=1.3.0',
        'numpy>=1.21.0',
        'tqdm>=4.62.0',
        'pytest>=7.0.0',
    ],
) 