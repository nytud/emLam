from functools import partial
import importlib
import inspect
import pkgutil


def get_all_corpora():
    """Returns all corpus classes defined in the corpus package."""
    from emLam.corpus.corpus_base import Corpus
    return get_all_classes(Corpus)


def get_all_preprocessors():
    """Returns all preprocessor classes defined in the corpus package."""
    from emLam.corpus.preprocessor_base import Preprocessor
    return get_all_classes(Preprocessor)


def get_all_classes(ancestor):
    """
    Returns all classes of a specific type defined in the corpus package. In
    order to support configuration that follows the class hierarchy, the data
    is returned as {component_name: component_class, [path_to_ancestor]}.
    """
    def is_mod_class(mod, cls):
        return inspect.isclass(cls) and inspect.getmodule(cls) == mod

    curr_module = importlib.import_module(__name__)
    classes = {}
    for _, module_name, _ in pkgutil.iter_modules(curr_module.__path__):
        module = importlib.import_module(__name__ + '.' + module_name)
        for _, cls in inspect.getmembers(module, partial(is_mod_class, module)):
            # Only take 'named' classes, i.e. leaves in the class tree
            if issubclass(cls, ancestor) and cls.name:
                path = inspect.getmro(cls)
                path = path[:path.index(ancestor) + 1]
                path = [c.name or c.__name__ for c in path][::-1]
                classes[cls.name] = cls, path
    return classes
