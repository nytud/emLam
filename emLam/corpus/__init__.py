from functools import partial
import importlib
import inspect
import os
import os.path as op


def get_all_corpora():
    """Returns all corpus classes defined in the corpus package."""
    from emLam.corpus.corpus_base import Corpus
    return get_all_classes(Corpus)


def get_all_preprocessors():
    """Returns all preprocessor classes defined in the corpus package."""
    from emLam.corpus.preprocessor_base import Preprocessor
    return get_all_classes(Preprocessor)


def get_all_classes(ancestor):
    """Returns all classes of a specific type defined in the corpus package."""
    def is_mod_class(mod, cls):
        return inspect.isclass(cls) and inspect.getmodule(cls) == mod

    classes = {}
    cdir = op.dirname(op.abspath(__file__))
    for cfile in filter(lambda f: f.endswith('.py'), os.listdir(cdir)):
        module_name = inspect.getmodulename(op.join(cdir, cfile))
        # TODO This is ugly; use ABC's or metaprogramming, see
        # http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html
        module = importlib.import_module(__name__ + '.' + module_name)
        for _, cls in inspect.getmembers(module, partial(is_mod_class, module)):
            # Only take 'named' classes, i.e. leaves in the class tree
            if issubclass(cls, ancestor) and cls.name:
                classes[cls.name] = cls
    return classes
