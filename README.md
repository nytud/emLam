# emLam

Preprocessing and modeling scripts for Hungarian Language Modeling

## Installation

The package can be installed with either of
```bash
pip install .
python setup.py install
```
(though [the former is preferred over the latter](http://stackoverflow.com/questions/15724093/)).
These commands install all packages required by the preprocessing scripts. In
order to use the RNN models, `tensorflow` and `numpy` must be installed
separately:
```bash
# For nVidia GPUs -- strongly recommended
pip install -r requirements_gpu.txt
# In every other case
pip install -r requirements.txt
```
