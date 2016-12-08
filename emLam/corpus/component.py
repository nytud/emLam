#!/usr/bin/env python3
"""An instantiable component. Also has bindings for ArgumentParser."""

from __future__ import absolute_import, division, print_function
import inspect


class NamedClass(type):
    """
    A read-only name property for classes. See
    http://stackoverflow.com/questions/3203286/how-to-create-a-read-only-class-property-in-python
    """
    @property
    @classmethod
    def name(cls):
        return getattr(cls, 'NAME', None)

class Component(object, metaclass=NamedClass):
    """Base class for corpus objects."""
    @classmethod
    def parser(cls, subparsers):
        """
        This method adds a (n ArgumentParser) subparser to the group specified
        in the argument.
        """
        raise NotImplementedError('parser() must be implemented')

    @classmethod
    def instantiate(cls, process_id=0, **kwargs):
        """
        Instantiates the class from keyword arguments. The process_id (not a
        real pid, but an ordinal starting from 0) is there so that components
        that use external resources can "plan" accordingly.
        """
        argspec = inspect.getargspec(cls.__init__).args
        component_args = {k: kwargs[k] for k in argspec[1:] if k in kwargs}
        return cls(**component_args)

    @classmethod
    def name(cls):
        """The display name of the component."""
        raise NotImplementedError('name() must be implemented')
