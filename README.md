# adaptivesswt
Adaptive Synchrosqueezing Wavelet Transform

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
>>> config = adaptivesswt.Configuration(minFreq=0.1, fmamaxFreq=10, numFreqs=20)
>>> sst, freqs, tail = adaptivesswt.adaptive_sswt(signal, batch_iters=10, method='proportional', thrsh=1/10, otl=False, **config.asdict())
```