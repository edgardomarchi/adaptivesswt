# adaptivesswt
Adaptive Synchrosqueezing Wavelet Transform

This Python package computes an adaptive version of the synchrosqueezing transform intended to improve resolution
of the time-frequency representatiton of a multi-component signal using very few analysis frequencies (wavelets).
It is aimed towards efficiency, implemented with multithreading and multiprocessing, it provides several configurations parameters to the user in order to be adaptable to a variety of requirements related with
real world applications.

It relies on the [```PyWavelets```](https://github.com/PyWavelets/pywt) package.

## Requirements
```python = ">=3.7,<3.11"```

Package dependencies will be automatically installed by ```pip```.

## Installation
For now:

```
$ pip install git+https://github.com/edgardomarchi/adaptivesswt.git#egg=adaptivesswt
```

## Usage
After importing the module it is recommended to create a configuration object to simplify passing parameters, for example:

```python
>>> import adaptivesswt
>>> import numpy as np
>>> signal = np.random.rand(2048)
>>> config = adaptivesswt.Configuration(min_freq=0.1, max_freq=10, num_freqs=20)
>>> sst, freqs, tail = adaptivesswt.adaptive_sswt(signal, batch_iters=10, method='proportional', thrsh=1/10, otl=False, **config.asdict())
```
## Examples
Examples and helper scripts can be found in the [scripts folder](scripts/). Check [SCRIPTS.md](scripts/SCRIPTS.md) for more info.

## License
```adaptivesswt``` is free and Open Source software released under GPL license.

## Acknowledgment
This work is product of a collaboration project between the **Communications Department** of the [_Instituto Nacional de Tecnología Industrial (INTI)_](http://www.inti.gob.ar) and the **Computational Simulation Center (CSC) for Technological Applications** of the [_Consejo Nacional de Investigaciones Científicas y Técnicas (CoNICET)_](http://www.conicet.gov.ar).

## Citing
