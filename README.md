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


## Further resources
```
The [emLam corpus](http://hlt.bme.hu/en/resources/emLam), a specially prepared
version of the [Hungarian Webcorpus](http://mokk.bme.hu/resources/webcorpus/),
is available from http://hlt.bme.hu/en/resources/emLam.

If you use the repository or the corpus in your project, please cite the following
paper:

Dávid Márk Nemeskey 2017. `emLam` – a Hungarian Language Modeling
baseline. In _Proceedings of the 13th Conference on Hungarian Computational
Linguistics (MSZNY 2017)_. [(bib here)](http://hlt.bme.hu/en/publ/emLam)
