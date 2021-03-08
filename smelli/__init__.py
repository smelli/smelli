from ._version import __version__, __flavio__version__
import flavio
from pkg_resources import packaging
_flavio_up_to_date = (
    packaging.version.parse(flavio.__version__)
    >= packaging.version.parse(__flavio__version__)
)
from . import classes
from . import util
from .classes import GlobalLikelihood
