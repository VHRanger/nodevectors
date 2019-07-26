from setuptools import setup, find_packages


setup(
    name="graph2vec",
    version="0.0.1",
    license='MIT',
    description='Fast networkx graph node embeddings',
    author='Matt Ranger',
    url='https://github.com/VHRanger/graph2vec/',
    packages=find_packages(),
    keywords=['graph', 'network', 'embedding', 'node2vec'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.txt', '*.rst']
    },
    install_requires=[
        'gensim',
        'numba',
        'numpy',
        'pandas',
        'scipy',
      ],
)
