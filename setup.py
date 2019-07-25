from setuptools import setup, find_packages


setup(
    name="EmbeddingLib",
    version="0.0.1",
    packages=find_packages(),

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.txt', '*.rst']
    },

    install_requires=[
        'gensim',
        'numpy',
        'pandas',
        'sklearn',
		'pytest',
		'pytest-cov'
      ],
)
