# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['adaptivesswt', 'adaptivesswt.utils']

package_data = \
{'': ['*']}

install_requires = \
['SoundFile>=0.10.3.post1,<0.11.0',
 'matplotlib',
 'numba>=0.55,<0.56',
 'numpy>=1.20,<2.0',
 'pywavelets>=1.2,<2.0',
 'scipy']

setup_kwargs = {
    'name': 'adaptivesswt',
    'version': '0.2.1',
    'description': 'A package to calculate an Adaptive Synchrosqueezing Transform',
    'long_description': '# adaptivesswt\nAdaptive Synchrosqueezing Wavelet Transform\n\nThis Python package computes an adaptive version of the synchrosqueezing transform intended to improve resolution\nof the time-frequency representatiton of a multi-component signal using very few analysis frequencies (wavelets).\nIt is aimed towards efficiency, implemented with multithreading and multiprocessing, it provides several configurations parameters to the user in order to be adaptable to a variety of requirements related with \nreal world applications.\n\nIt relies on the [```PyWavelets```](https://github.com/PyWavelets/pywt) package.\n\n## Requirements\n```python = ">=3.7,<3.11"```\n\nPackage dependencies will be automatically installed by ```pip```.\n\n## Installation\nFor now:\n\n```\n$ pip install git+https://github.com/edgardomarchi/adaptivesswt.git#egg=adaptivesswt\n```\n\n## Usage\nAfter importing the module it is recommended to create a configuration object to simplify passing parameters, for example:\n\n```python\n>>> import adaptivesswt\n>>> import numpy as np\n>>> signal = np.random.rand(2048)\n>>> config = adaptivesswt.Configuration(min_freq=0.1, max_freq=10, num_freqs=20)\n>>> sst, freqs, tail = adaptivesswt.adaptive_sswt(signal, batch_iters=10, method=\'proportional\', thrsh=1/10, otl=False, **config.asdict())\n```\n## Examples\nExamples and helper scripts can be found in the [scripts folder](scripts/). Check [SCRIPTS.md](scripts/SCRIPTS.md) for more info.\n\n## License\n```adaptivesswt``` is free and Open Source software released under GPL license.\n\n## Acknowledgment\nThis work is product of a collaboration project between the **Communications Department** of the [_Instituto Nacional de Tecnología Industrial (INTI)_](http://www.inti.gob.ar) and the **Computational Simulation Center (CSC) for Technological Applications** of the [_Consejo Nacional de Investigaciones Científicas y Técnicas (CoNICET)_](http://www.conicet.gov.ar).\n\n## Citing\n',
    'author': 'Edgardo Marchi',
    'author_email': 'emarchi@inti.gob.ar',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/edgardomarchi/adaptivesswt',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<3.11',
}


setup(**setup_kwargs)
