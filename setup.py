import os
from pathlib import Path
from setuptools import setup, find_packages
import tomllib

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

with open(PROJECT_DIR / "pyproject.toml", "rb") as f:
    tm = tomllib.load(f)['project']

print(tm)
    
setup(
    name=tm['name'],
    version=tm['version'],
    license=tm['license'],
    author=tm['authors'][0]['name'],
    url=tm['urls']['github'],
    packages=find_packages(),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.txt', '*.rst']
    },
    install_requires=tm['dependencies'],
)
