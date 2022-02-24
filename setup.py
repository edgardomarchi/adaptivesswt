# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['adaptivesswt', 'adaptivesswt.utils']

package_data = \
{'': ['*']}

install_requires = \
['numba', 'numpy>=1.20,<2.0', 'pywavelets>=1.2,<2.0', 'scipy']

setup_kwargs = {
    'name': 'adaptivesswt',
    'version': '0.1.0',
    'description': 'A package to calculate an Adaptive Synchrosqueezing Transform',
    'long_description': '# adaptivesswt\nAdaptive Synchrosqueezing Wavelet Transform\n\n## Installation\nFor now:\n\n```\n$ pip install git+https://github.com/edgardomarchi/adaptivesswt.git#egg=adaptivesswt\n```',
    'author': 'Edgardo Marchi',
    'author_email': 'emarchi@inti.gob.ar',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/edgardomarchi/adaptivesswt',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
