from functools import partial
import importlib
import inspect
import os
import os.path as op
import re


__corpus_p = re.compile(r'(.+_.+)[.]py')


def get_all_corpora():
    def is_mod_class(mod, cls):
        return inspect.isclass(cls) and inspect.getmodule(cls) == mod

    corpora = {}

    cdir = op.dirname(op.abspath(__file__))
    for cfile in os.listdir(cdir):
        m = __corpus_p.match(cfile)
        module_name = inspect.getmodulename(op.join(cdir, cfile))
        if module_name and module_name != 'corpus_base':
            module = importlib.import_module(__name__ + '.' + module_name)
            for cls in inspect.getmembers(module, partial(is_mod_class, module)):
                corpora[module_name] = cls[1]
    return corpora
