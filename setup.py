from setuptools import setup, find_packages

setup(
    name='scNET',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2.1',
        'torch-cluster==1.6.3'
        'torch-geometric==2.1.0.post1'
        'torch-scatter==2.1.2'
        'torch-sparse ==0.6.18',
        'pandas>=2.2.1',
        'numpy>=1.26.4'
        'networkx>=3.1',
        'scanpy>=1.9.8',
        'scikit-learn>=1.4.1',
        'gseapy>=1.1.2',
        'matplotlib>=3.8.0'
    ],
    author='Ron Sheinin',
    description='Our method employs a unique dual-graph architecture based on graph neural networks (GNNs), enabling the joint representation of gene expression and PPI network data',
    url='https://github.com/madilabcode/scNET'
)