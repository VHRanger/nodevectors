from setuptools import setup, find_packages

setup(
    name="graph2vec-learn",
    version="0.1.5",
    license='MIT',
    description='Fast networkx graph node embeddings',
    author='Matt Ranger',
    url='https://github.com/VHRanger/graph2vec/',
    packages=find_packages(),
    download_url='https://github.com/VHRanger/graph2vec/archive/0.0.1.tar.gz',
    keywords=['graph', 'network', 'embedding', 'node2vec'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.txt', '*.rst']
    },
    install_requires=[
    'CSRGraph',
    'gensim',
    'networkx',
    'numba',
    'numpy',
    'pandas',
    'scipy',
    ],
)
