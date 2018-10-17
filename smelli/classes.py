import flavio
from wilson import Wilson
import wcxf
from flavio.statistics.likelihood import Likelihood, FastLikelihood
from flavio.statistics.probability import NormalDistribution
from flavio.statistics.functions import pull
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
from math import ceil
from .util import tree, get_datapath
from multipledispatch import dispatch
import os


# by default, smelli uses leading log accuracy for SMEFT running!
Wilson.set_default_option('smeft_accuracy', 'leadinglog')



class GlobalLikelihood(object):
    """Class that provides a global likelihood in SMEFT Wilson
    coefficient space.

    User methods:

    - `log_likelihood`: return an instance of LieklihoodResult
    given a dictionary of Wilson coefficients at a given scale
    - `log_likelihood_wcxf`: return an instance of LieklihoodResult
    given the path to a WCxf file
    - `log_likelihood_wilson`: return an instance of LieklihoodResult+
    given an instance of `wilson.Wilson`

    Utility methods:

    - `make_measurement`: compute the SM covariances. Note that it is only
    necessary to call this method when changes to the default
    parameters/uncertainties have been made
    - `save_sm_covariances`, `load_sm_covariances`: Save the calculated SM
    covariances or load them from data files

    """

    _default_bases = {'SMEFT': 'Warsaw', 'WET': 'flavio'}

    def __init__(self, eft='SMEFT', basis=None, par_dict=None,
                 include_likelihoods=None,
                 exclude_likelihoods=None,
                 Nexp=5000):
        """Initialize the likelihood.

        Optionally, a dictionary of parameters can be passed as `par_dict`.
        If not given (or not complete), flavio default parameter values will
        be used.
        """
        self.eft = eft
        self.basis = basis or self._default_bases[self.eft]
        par_dict = par_dict or {}  # initialize empty if not given
        # take missing parameters from flavio defaults
        self.par_dict = flavio.default_parameters.get_central_all()
        self.par_dict.update(par_dict)
        self.likelihoods = {}
        self.fast_likelihoods = {}
        self._load_likelihoods(include_likelihoods=include_likelihoods,
                               exclude_likelihoods=exclude_likelihoods)
        try:
            self.load_sm_covariances(get_datapath('smelli', 'data/cache'))
            self.make_measurement(Nexp=Nexp)
        except FileNotFoundError:
            warnings.warn("The cached SM covariances were not found. "
                          "Please call `make_measurement` to compute them.")
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            warnings.warn("There was a problem loading the SM covariances. "
                          "Please recompute them with `make_measurement`.")
        self._log_likelihood_sm = None

    def _load_likelihoods(self,
                          include_likelihoods=None,
                          exclude_likelihoods=None):
        if include_likelihoods is not None and exclude_likelihoods is not None:
            raise ValueError("include_likelihoods and exclude_likelihoods "
                             "should not be specified simultaneously.")
        for fn in ['fast_likelihood_quarks.yaml',
                   'fast_likelihood_leptons.yaml']:
            if include_likelihoods is not None and fn not in include_likelihoods:
                continue
            if exclude_likelihoods is not None and fn in exclude_likelihoods:
                continue
            with open(get_datapath('smelli',
                                   'data/yaml/' + fn), 'r') as f:
                L = FastLikelihood.load(f)
            self.fast_likelihoods[fn] = L
        for fn in ['likelihood_ewpt.yaml',
                   'likelihood_lept.yaml',
                   'likelihood_rd_rds.yaml',
                   'likelihood_lfu_fccc.yaml',
                   'likelihood_lfu_fcnc.yaml',
                   'likelihood_bcpv.yaml',
                   'likelihood_bqnunu.yaml',
                   'likelihood_lfv.yaml',
                   'likelihood_zlfv.yaml',
                   ]:
            if include_likelihoods is not None and fn not in include_likelihoods:
                continue
            if exclude_likelihoods is not None and fn in exclude_likelihoods:
                continue
            if self.eft != 'SMEFT' and fn in ['likelihood_ewpt.yaml',
                                              'likelihood_zlfv.yaml',]:
                continue
            with open(get_datapath('smelli',
                                   'data/yaml/' + fn), 'r') as f:
                L = Likelihood.load(f)
            self.likelihoods[fn] = L

    def make_measurement(self, *args, **kwargs):
        for name, flh in self.fast_likelihoods.items():
            flh.make_measurement(*args, **kwargs)

    def save_sm_covariances(self, folder):
        for name, flh in self.fast_likelihoods.items():
            filename = os.path.join(folder, name + '.p')
            flh.sm_covariance.save(filename)

    def load_sm_covariances(self, folder):
        for name, flh in self.fast_likelihoods.items():
            filename = os.path.join(folder, name + '.p')
            flh.sm_covariance.load(filename)

    @property
    def log_likelihood_sm(self):
        if self._log_likelihood_sm is None:
            self._log_likelihood_sm = self._log_likelihood(flavio.WilsonCoefficients())
        return self._log_likelihood_sm

    def get_wilson(self, wc_dict, scale):
        return Wilson(wc_dict, scale=scale, eft=self.eft, basis=self.basis)

    def _log_likelihood(self, w):
        """Return the log-likelihood as a dictionary for an instance of
        `wilson.Wilson`."""
        ll = {}
        for name, flh in self.fast_likelihoods.items():
            ll[name] = flh.log_likelihood(self.par_dict, w)
        for name, lh in self.likelihoods.items():
            ll[name] = lh.log_likelihood(self.par_dict, w)
        return ll

    @dispatch(dict, (int, float))
    def parameter_point(self, wc_dict, scale):
        """Choose a point in parameter space by providing a dictionary of
        Wilson coefficient values (with keys corresponding to WCxf Wilson
        coefficient names) and the input scale."""
        w = self.get_wilson(wc_dict, scale)
        return GlobalLikelihoodPoint(self, w)

    @dispatch(str)
    def parameter_point(self, filename):
        """Choose a point in parameter space by providing the path to a WCxf
        file."""
        with open(filename, 'r') as f:
            wc = wcxf.WC.load(f)
        w = Wilson.from_wc(wc)
        return GlobalLikelihoodPoint(self, w)

    @dispatch(Wilson)
    def parameter_point(self, w):
        """Choose a point in parameter space by providing an instance
        of `wilson.Wilson`."""
        return GlobalLikelihoodPoint(self, w)


class GlobalLikelihoodPoint(object):
    """Class representing the properties of the likelihood function at a
    specific point in parameter space.

    Attributes:

    - `log_likelihood_dict`: dictionary with individual contributions
    to the log-likelihood
    - `value`: Return the numerical values of the global log-likelihood
    compared to the SM value (can also be acessed with `float(self)`)

    Methods:

    - `get_obstable`: return a pandas data frame with the values and pulls
    for each individual observable, given the Wilson coefficients
    """

    def __init__(self, likelihood, w):
        self.likelihood = likelihood
        self.w = w
        self._obstable_tree_cache = None
        self._log_likelihood_dict = None

    def _delta_log_likelihood(self):
        """Compute the delta log likelihood for the individual likelihoods"""
        ll = self.likelihood._log_likelihood(self.w)
        for name in ll:
            ll[name] -= self.likelihood.log_likelihood_sm[name]
        ll['global'] = sum(ll.values())
        return ll

    def log_likelihood_dict(self):
        """Return a dictionary with the delta log likelihood values
        for the individual contributions.

        Cached after the first call."""
        if self._log_likelihood_dict is None:
            self._log_likelihood_dict = self._delta_log_likelihood()
        return self._log_likelihood_dict

    def log_likelihood_global(self):
        """Return the value of the global delta log likelihood.

        Cached after the first call. Corresponds to the `global` key of
        the dictionary returned by `log_likelihood_dict`."""
        return self.log_likelihood_dict()['global']

    @staticmethod
    def _get_inspire_dict(observables, ml):
        inspire_dict = {}
        obs_set = set(observables)
        for m_name in ml.get_measurements:
            m_obj = flavio.Measurement[m_name]
            for obs in set(m_obj.all_parameters) & obs_set:
                if obs in inspire_dict:
                    inspire_dict[obs].append(m_obj.inspire)
                else:
                    inspire_dict[obs]=[m_obj.inspire]
        return inspire_dict

    @property
    def _obstable_tree(self):
        if not self._obstable_tree_cache:
            info = tree()  # nested dict
            pull_dof = 1
            llh = self.likelihood
            for flh_name, flh in llh.fast_likelihoods.items():
                # loop over fast likelihoods: they only have a single "measurement"
                m = flh.pseudo_measurement
                ml = flh.full_measurement_likelihood
                pred = ml.get_predictions_par(llh.par_dict, self.w)
                sm_cov = flh.sm_covariance.get(force=False)
                _, exp_cov = flh.exp_covariance.get(force=False)
                inspire_dict = self._get_inspire_dict(flh.observables, ml)
                for i, obs in enumerate(flh.observables):
                    info[obs]['lh_name'] = flh_name
                    info[obs]['name'] = obs if isinstance(obs, str) else obs[0]
                    info[obs]['theory'] = pred[obs]
                    info[obs]['th. unc.'] = np.sqrt(sm_cov[i, i])
                    info[obs]['experiment'] = m.get_central(obs)
                    info[obs]['exp. unc.'] = np.sqrt(exp_cov[i, i])
                    info[obs]['exp. PDF'] = NormalDistribution(m.get_central(obs), np.sqrt(exp_cov[i, i]))
                    info[obs]['inspire'] = sorted(set(inspire_dict[obs]))
                    ll_central = m.get_logprobability_single(obs, m.get_central(obs))
                    ll = m.get_logprobability_single(obs, pred[obs])
                    # DeltaChi2 is -2*DeltaLogLikelihood
                    info[obs]['pull'] = pull(-2 * (ll - ll_central), dof=pull_dof)
            for lh_name, lh in llh.likelihoods.items():
                # loop over "normal" likelihoods
                ml = lh.measurement_likelihood
                pred = ml.get_predictions_par(llh.par_dict, self.w)
                inspire_dict = self._get_inspire_dict(lh.observables, ml)
                for i, obs in enumerate(lh.observables):
                    obs_dict = flavio.Observable.argument_format(obs, 'dict')
                    obs_name = obs_dict.pop('name')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p_comb = flavio.combine_measurements(
                            obs_name,
                            include_measurements=ml.get_measurements,
                            **obs_dict)
                    info[obs]['experiment'] = p_comb.central_value
                    info[obs]['exp. unc.'] = max(p_comb.error_left, p_comb.error_right)
                    info[obs]['exp. PDF'] = p_comb
                    info[obs]['inspire'] = sorted(set(inspire_dict[obs]))
                    info[obs]['theory'] = pred[obs]
                    info[obs]['th. unc.'] = 0
                    info[obs]['lh_name'] = lh_name
                    info[obs]['name'] = obs if isinstance(obs, str) else obs[0]
                    ll = p_comb.logpdf([pred[obs]]) - p_comb.logpdf([p_comb.central_value])
                    info[obs]['pull'] = pull(-2 * ll, dof=pull_dof)
            self._obstable_tree_cache = info
        return self._obstable_tree_cache

    def obstable(self, min_pull=0):
        r"""Return a pandas data frame with the central values and uncertainties
        as well as the pull for each observable.

        The pull is defined is $\sqrt(|-2\ln L|)$. Note that the global
        likelihood is *not* simply proportional to the sum of squared pulls
        due to correlations.
        """
        info = self._obstable_tree
        info = self._obstable_filter_sort(info, min_pull=min_pull,
                                          sortkey='pull', reverse=True)
        # create DataFrame
        df = pd.DataFrame(info).T
        # remove inspire references, likelihood name, and observable name
        del(df['inspire'])
        del(df['lh_name'])
        del(df['name'])
        del(df['exp. PDF'])
        return df

    @staticmethod
    def _obstable_filter_sort(info, min_pull=0, max_pull=np.inf, sortkey='name', reverse=False, subset=None, max_rows=None):
        # impose min_pull and max_pull
        info = {obs:row for obs,row in info.items()
                if row['pull'] >= min_pull and row['pull'] <= max_pull}
        # get only subset:
        if subset is not None:
            info = {obs:row for obs,row in info.items() if subset(row)}
        # sort
        info = OrderedDict(sorted(info.items(), key=lambda x: x[1][sortkey],
                                  reverse=reverse))
        # restrict number of rows per tabular to max_rows
        if max_rows is None or len(info)<=max_rows:
            return info
        else:
            info_list = []
            for n in range(ceil(len(info)/max_rows)):
                info_n = OrderedDict((obs,row)
                                    for i,(obs,row) in enumerate(info.items())
                                    if i>=n*max_rows and i<(n+1)*max_rows)
                info_list.append(info_n)
            return info_list
