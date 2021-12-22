# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['pysoc']

package_data = {'': ['*']}

setup_kwargs = {
    'name': 'pysoc',
    'version': '0.1.0',
    'description': 'Social Choice Theory (SCT) library.',
    'long_description': None,
    'author': 'Jeremy Silver',
    'author_email': 'jeremys@nessiness.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)

