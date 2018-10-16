import pkgutil
import os
import sys
from collections import defaultdict


def tree():
    """Tree data structure.

    See https://gist.github.com/hrldcpr/2012250
    """
    return defaultdict(tree)


def get_datapath(package, resource):
    """Rewrite of pkgutil.get_data() that just returns the file path.

    Taken from https://stackoverflow.com/a/13773912"""
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None
    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)
    return resource_name
