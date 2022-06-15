# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['adaptivesswt', 'adaptivesswt.utils']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib', 'numba', 'numpy>=1.20,<2.0', 'pywavelets>=1.2,<2.0', 'scipy']

setup_kwargs = {
    'name': 'adaptivesswt',
    'version': '0.1.1',
    'description': 'A package to calculate an Adaptive Synchrosqueezing Transform',
    'long_description': "# adaptivesswt\nAdaptive Synchrosqueezing Wavelet Transform\n\n## Installation\nFor now:\n\n```\n$ pip install git+https://github.com/edgardomarchi/adaptivesswt.git#egg=adaptivesswt\n```\n\n## Usage\nAfter importing the module it is recommended to create a configuration object to simplify passing parameters, for example:\n\n```python\n>>> import adaptivesswt\n>>> import numpy as np\n>>> signal = np.random.rand(2048)\n>>> config = adaptivesswt.Configuration(minFreq=0.1, fmamaxFreq=10, numFreqs=20)\n>>> sst, freqs, tail = adaptivesswt.adaptive_sswt(signal, batch_iters=10, method='proportional', thrsh=1/10, otl=False, **config.asdict())\n```",
    'author': 'Edgardo Marchi',
    'author_email': 'emarchi@inti.gob.ar',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/edgardomarchi/adaptivesswt',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
