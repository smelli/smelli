from collections import OrderedDict
from flavio.io import instanceio as iio
import numpy as np
import yaml
from .util import get_cachepath
from requests import get
import os
from wilson import Wilson
from sklearn.neighbors import KernelDensity
import wcxf
from itertools import chain

class UnbinnedParameterLikelihood(iio.YAMLLoadable):
    """An `UnbinnedParameterLikelihood` provides an unbinned likelihood function in terms of
    parameters.
    Methods:
    - `log_likelihood`: The likelihood as a function of the parameters
    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """
    _input_schema_dict = {
        'eft': str,
        'basis': str,
        'scale': float,
        'names': list,
        'samples': str,
        'weights': str
    }

    _output_schema_dict = {
        'eft': str,
        'basis': str,
        'scale': float,
        'names': list,
        'samples': str,
        'weights': str
    }

    def _make_real_acc(i):
        return lambda wcs: np.real(wcs[i])

    def _make_imag_acc(i):
        return lambda wcs: np.imag(wcs[i])

    def _complex_to_real(self, values):
        return list(chain.from_iterable(
            [np.real(values[i])] if 'real' in self._basis_wcs[n]
            else
            [np.real(values[i]), np.imag(values[i])]
            for i, n in enumerate(self.names)
        ))

    def __init__(self,
                 eft,
                 basis,
                 scale,
                 names,
                 samples,
                 base_path,
                 weights=None):
        """Initialize the instance.
        Parameters:
        - `eft`: a string identifying the WET as defined by wcxf
        - `basis`: a string identifying the WET basis as defined by wcxf
        - `scale`: a scale (in units of GeV) at which the WET coefficient samples are valid
        - `names`: an ordered list of names identiying the Wilson coefficients in the WET basis
        - `samples`: a numpy array of vectors of Wilson coefficients, with each row representing a sample
        - `weights`: a numpy array of scalar weights on a logarithmic scale for each sample
        """
        self.eft     = eft
        self.basis   = basis
        self.scale   = scale
        self.names   = names
        self.samples = np.load(os.path.join(base_path, samples))
        self.weights = np.ones(len(self.samples)) if weights is None else np.load(os.path.join(base_path, weights))
        self.kde     = KernelDensity(algorithm='auto', kernel='gaussian', rtol=1e-4)

        basis = wcxf.Basis[eft, basis]
        self._basis_wcs = {}
        for sector in basis.sectors.values():
            self._basis_wcs.update(sector)

        self.samples = np.array([
            self._complex_to_real(row) for row in self.samples
        ])
        self.kde.fit(X=self.samples, sample_weight=self.weights)

    def log_likelihood(self, par_dict, w):
        """Return the normalized log-likelihood for all parameters."""
        if type(w) is Wilson:
            smeft_wc = w
        else:
            smeft_wc = Wilson.from_wc(w.get_wcxf(sector='all', scale=self.scale, par=par_dict, eft='SMEFT', basis='Warsaw'))
        wet_wc = smeft_wc.match_run(self.scale, self.eft, self.basis)
        values = _complex_to_real([wet_wc[name] for name in self.names])
        return self.kde.score_samples(X=[values])[0]

    def test_perplexity(self):
        test_kde = KernelDensity(algorithm='auto', kernel='gaussian', rtol=1e-4)
        test_kde.fit(X=self.samples[:len(self.samples)//2], sample_weight=self.weights[:len(self.samples)//2])
        lnw = test_kde.score_samples(X=self.samples[len(self.samples)//2:])
        w = np.exp(lnw)
        return np.exp(np.dot(w, lnw)/len(lnw))

    @classmethod
    def load(cls, f, **kwargs):
        """Instantiate an object from a YAML string or stream."""
        d = yaml.load(f, Loader=yaml.Loader)
        return cls.load_dict(d, **kwargs)

    def get_yaml_dict(self):
        """Dump the object to a YAML dictionary."""
        d = self.__dict__.copy()
        schema = self.output_schema()
        d = schema(d)
        # remove NoneTypes and empty lists
        d = {k: v for k, v in d.items() if v is not None}
        return d

class DataSets(iio.YAMLLoadable):
    """`DataSets` provides an interface to the list of data sets used for unbinned likelihood functions.
    Methods:
    - `log_likelihood`: The likelihood as a function of the parameters
    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """
    _input_schema_dict = {
        'datasets': dict,
    }

    _output_schema_dict = {
        'datasets': dict,
    }

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, dn):
        """Return a description of a known dataset by key, or None."""
        if not dn in self.datasets.keys():
            return None

        return datasets[dn]

    def __iter__(self):
        yield from self.datasets.keys()

    @classmethod
    def load(cls, f, **kwargs):
        """Instantiate an object from a YAML string or stream."""
        d = yaml.load(f, Loader=yaml.Loader)
        return cls.load_dict(d, **kwargs)

    def download(self, dn):
        """Download a dataset to the cache."""
        if dn not in self.datasets.keys():
            raise ValueError('unknown dataset {}'.format(dn))

        destdir = os.path.join(get_cachepath(), dn)
        if not os.path.exists(destdir):
            os.makedirs(destdir)

        for url in self.datasets[dn]['urls']:
            fn = os.path.join(destdir, url[url.rfind('/') + 1:])
            content = get(url).content
            with open(fn, 'wb+') as f:
                f.write(content)
