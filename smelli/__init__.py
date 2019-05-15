from ._version import __version__, __flavio__version__
from . import classes
from . import util
from .classes import GlobalLikelihood
import flavio
import warnings


def _check_flavio_version():
    version_installed = flavio.__version__.split('.')
    version_required = __flavio__version__.split('.')
    if version_installed[:2] == version_required[:2]:
        return True
    else:
        return False


if not _check_flavio_version():
    warnings.warn("Your installed flavio version ({})"
                  .format(flavio.__version__)
                  + " is not compatible with this version of smelli."
                  " Please check for available updates.")
