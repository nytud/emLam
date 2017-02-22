# lstm_lm.py

## Arguments

The script reads its arguments from two sources: the command line and a
configuration file; the latter must be specified in the former with the `-c`
option. Arguments pertaining to the network structure, training hyperparameters,
etc. are specified in the configuration file, while
inputs, logging, etc. on the command line. This makes it possible to re-use the
same configuration file for various input corpora.

As usual, the command line parameters are explained in the script's usage
description (`-h`). The options in the configuration file can be found in
[the schema](conf/lstm_lm_conf.schema).
