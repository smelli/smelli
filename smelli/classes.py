import flavio
from wilson import Wilson
import wcxf
from flavio.statistics.likelihood import Likelihood, FastLikelihood
from flavio.statistics.probability import NormalDistribution
from flavio.statistics.functions import pull, pvalue
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
from math import ceil
from .util import tree, get_datapath
from multipledispatch import dispatch
from copy import copy
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
    - `save_exp_covariances`, `load_exp_covariances`: Save the calculated
    experimental central values and covariances or load them from data files

    """

    _default_bases = {'SMEFT': 'Warsaw', 'WET': 'flavio'}

    _fast_likelihoods_yaml = [
        'fast_likelihood_quarks.yaml',
        'fast_likelihood_leptons.yaml'
    ]

    _likelihoods_yaml = [
        'likelihood_ewpt.yaml',
        'likelihood_lept.yaml',
        'likelihood_rd_rds.yaml',
        'likelihood_lfu_fccc.yaml',
        'likelihood_lfu_fcnc.yaml',
        'likelihood_bcpv.yaml',
        'likelihood_bqnunu.yaml',
        'likelihood_lfv.yaml',
        'likelihood_zlfv.yaml',
    ]

    def __init__(self, eft='SMEFT', basis=None, par_dict=None,
                 include_likelihoods=None,
                 exclude_likelihoods=None,
                 Nexp=5000,
                 exp_cov_folder=None,
                 sm_cov_folder=None,
                 custom_likelihoods=None):
        """Initialize the likelihood.

        Optionally, a dictionary of parameters can be passed as `par_dict`.
        If not given (or not complete), flavio default parameter values will
        be used.


        Parameters:

        - eft: a WCxf EFT, must be one of 'SMEFT' (default) or 'WET'.
        - basis: a WCxf basis, defaults to 'Warsaw' for SMEFT and 'flavio'
          for WET.
        - include_likelihoods: a list of strings specifying the likelihoods
          to be included (default: all of them). Note that this cannot be used
          to add likelihoods.
        - exclude_likelihoods: a list of strings specifying the likelihoods
          to be excluded (default: none of them).
        - Nexp: number of random evaluations of the experimental likelihood
          used to extract the covariance matrix for "fast likelihood"
          instances. Defaults to 5000.
        - exp_cov_folder: directory containing saved expererimental
          covariances. The data files have to be in the format exported by
          `save_exp_covariances`.
        - sm_cov_folder: directory containing saved SM
          covariances. The data files have to be in the format exported by
          `save_sm_covariances`.
        - custom_likelihoods: a dictionary in which each value is a list of
          observables and each key is a string that serves as user-defined
          name. For each item of the dictionary, a custom likelihood will be
          computed.
        """
        self.eft = eft
        self.basis = basis or self._default_bases[self.eft]
        par_dict = par_dict or {}  # initialize empty if not given
        # take missing parameters from flavio defaults
        self.par_dict = flavio.default_parameters.get_central_all()
        self.par_dict.update(par_dict)
        self.likelihoods = {}
        self.fast_likelihoods = {}
        self._custom_likelihoods_dict = custom_likelihoods or {}
        self.custom_likelihoods = {}
        self._load_likelihoods(include_likelihoods=include_likelihoods,
                               exclude_likelihoods=exclude_likelihoods)
        self._Nexp = Nexp
        if exp_cov_folder is not None:
            self.load_exp_covariances(exp_cov_folder)
        self._sm_cov_loaded = False
        try:
            if sm_cov_folder is None:
                self.load_sm_covariances(get_datapath('smelli', 'data/cache'))
            else:
                self.load_sm_covariances(sm_cov_folder)
            self._sm_cov_loaded = True
            self.make_measurement()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            warnings.warn("There was a problem loading the SM covariances. "
                          "Please recompute them with `make_measurement`.")
        self._log_likelihood_sm = None
        self._obstable_sm = None

    def _load_likelihoods(self,
                          include_likelihoods=None,
                          exclude_likelihoods=None):
        if include_likelihoods is not None and exclude_likelihoods is not None:
            raise ValueError("include_likelihoods and exclude_likelihoods "
                             "should not be specified simultaneously.")
        for fn in self._fast_likelihoods_yaml:
            if include_likelihoods is not None and fn not in include_likelihoods:
                continue
            if exclude_likelihoods is not None and fn in exclude_likelihoods:
                continue
            with open(self._get_likelihood_path(fn), 'r') as f:
                L = FastLikelihood.load(f)
            self.fast_likelihoods[fn] = L
        for fn in self._likelihoods_yaml:
            if include_likelihoods is not None and fn not in include_likelihoods:
                continue
            if exclude_likelihoods is not None and fn in exclude_likelihoods:
                continue
            if self.eft != 'SMEFT' and fn in ['likelihood_ewpt.yaml',
                                              'likelihood_zlfv.yaml',]:
                continue
            with open(self._get_likelihood_path(fn), 'r') as f:
                L = Likelihood.load(f)
            self.likelihoods[fn] = L
        for name, observables in self._custom_likelihoods_dict.items():
            L = CustomLikelihood(self, name, observables)
            self.custom_likelihoods['custom_' + name] = L

    def _get_likelihood_path(self, name):
        """Return a path for the likelihood specified by `name`.
        If a YAML file with that name is found in the package's data
        directory, that is used. Otherwise, `name` is assumed to be a path.

        Raises `FileNotFoundError` if path does not exists.
        """
        path = get_datapath('smelli', 'data/yaml/' + name)
        if os.path.exists(path):
            return path
        path = get_datapath('smelli', 'data/yaml/' + name + '.yaml')
        if os.path.exists(path):
            return path
        if os.path.exists(name):
            return name
        if os.path.exists(name + '.yaml'):
            return name + '.yaml'
        else:
            raise FileNotFoundError("Likelihood YAML file '{}' was not found".format(name))

    def make_measurement(self, *args, **kwargs):
        """Initialize the likelihood by producing a pseudo-measurement containing both
        experimental uncertainties as well as theory uncertainties stemming
        from nuisance parameters.

        Optional parameters:

        - `N`: number of random computations for the SM covariance (computing
          time is proportional to it; more means less random fluctuations.)
        - `Nexp`: number of random computations for the experimental covariance.
          This is much less expensive than the theory covariance, so a large
          number can be afforded (default: 5000).
        - `threads`: number of parallel threads for the SM
          covariance computation. Defaults to 1 (no parallelization).
        - `force`: if True, will recompute SM covariance even if it
          already has been computed. Defaults to False.
        - `force_exp`: if True, will recompute experimental central values and
          covariance even if they have already been computed. Defaults to False.
        """
        if 'Nexp' not in kwargs:
            kwargs['Nexp'] = self._Nexp
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

    def save_exp_covariances(self, folder):
        for name, flh in self.fast_likelihoods.items():
            filename = os.path.join(folder, name + '.p')
            flh.exp_covariance.save(filename)

    def load_exp_covariances(self, folder):
        for name, flh in self.fast_likelihoods.items():
            filename = os.path.join(folder, name + '.p')
            flh.exp_covariance.load(filename)

    @property
    def log_likelihood_sm(self):
        if self._log_likelihood_sm is None:
            self._log_likelihood_sm = self._log_likelihood(flavio.WilsonCoefficients())
        return self._log_likelihood_sm

    def _check_sm_cov_loaded(self):
        """Check if the SM covariances have been computed or loaded."""
        if not self._sm_cov_loaded:
            raise ValueError("Please load or compute the SM covariances first"
                             " by calling `make_measurement`.")

    @property
    def obstable_sm(self):
        self._check_sm_cov_loaded()
        if self._obstable_sm is None:
            info = tree()  # nested dict
            for flh_name, flh in self.fast_likelihoods.items():
                # loop over fast likelihoods: they only have a single "measurement"
                m = flh.pseudo_measurement
                ml = flh.likelihood.measurement_likelihood
                pred_sm = ml.get_predictions_par(self.par_dict,
                                                flavio.WilsonCoefficients())
                sm_cov = flh.sm_covariance.get(force=False)
                _, exp_cov = flh.exp_covariance.get(force=False)
                inspire_dict = self._get_inspire_dict(flh.observables, ml)
                for i, obs in enumerate(flh.observables):
                    info[obs]['lh_name'] = flh_name
                    info[obs]['name'] = obs if isinstance(obs, str) else obs[0]
                    info[obs]['th. unc.'] = np.sqrt(sm_cov[i, i])
                    info[obs]['experiment'] = m.get_central(obs)
                    info[obs]['exp. unc.'] = np.sqrt(exp_cov[i, i])
                    info[obs]['exp. PDF'] = NormalDistribution(m.get_central(obs), np.sqrt(exp_cov[i, i]))
                    info[obs]['inspire'] = sorted(set(inspire_dict[obs]))
                    info[obs]['ll_sm'] = m.get_logprobability_single(obs, pred_sm[obs])
                    info[obs]['ll_central'] = m.get_logprobability_single(obs, m.get_central(obs))
            for lh_name, lh in self.likelihoods.items():
                # loop over "normal" likelihoods
                ml = lh.measurement_likelihood
                pred_sm = ml.get_predictions_par(self.par_dict,
                                                flavio.WilsonCoefficients())
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
                    info[obs]['th. unc.'] = 0
                    info[obs]['lh_name'] = lh_name
                    info[obs]['name'] = obs if isinstance(obs, str) else obs[0]
                    info[obs]['ll_sm'] = p_comb.logpdf([pred_sm[obs]])
                    info[obs]['ll_central'] = p_comb.logpdf([p_comb.central_value])
            self._obstable_sm = info
        return self._obstable_sm

    def get_wilson(self, wc_dict, scale):
        return Wilson(wc_dict, scale=scale, eft=self.eft, basis=self.basis)

    def _log_likelihood(self, w):
        """Return the log-likelihood as a dictionary for an instance of
        `wilson.Wilson`."""
        ll = {}
        for name, flh in self.fast_likelihoods.items():
            ll[name] = flh.log_likelihood(self.par_dict, w, delta=True)
        for name, lh in self.likelihoods.items():
            ll[name] = lh.log_likelihood(self.par_dict, w, delta=True)
        for name, clh in self.custom_likelihoods.items():
            ll[name] = clh.log_likelihood(self.par_dict, w, delta=True)
        return ll

    @dispatch(dict)
    def parameter_point(self, wc_dict, scale=None):
        """Choose a point in parameter space by providing a dictionary of
        Wilson coefficient values (with keys corresponding to WCxf Wilson
        coefficient names) and the input scale."""
        if not scale:
            raise ValueError("You need to provide a scale")
        w = self.get_wilson(wc_dict, scale)
        return GlobalLikelihoodPoint(self, w)

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

    def number_observations_dict(self, exclude_observables=None):
        nobs_dict = {}
        for name, flh in self.fast_likelihoods.items():
            nobs_dict[name] = len(set(flh.observables) - set(exclude_observables or []))
        for name, lh in self.likelihoods.items():
            ml =  lh.measurement_likelihood
            nobs_dict[name] = ml.get_number_observations(
                exclude_observables=exclude_observables
            )
        for name, clh in self.custom_likelihoods.items():
            nobs_dict[name] = clh.get_number_observations()
        nobs_dict['global'] = sum([v for k, v in nobs_dict.items() if 'custom_' not in k])
        return nobs_dict


class CustomLikelihood(object):
    def __init__(self, likelihood, name, observables):
        self.likelihood = likelihood
        self.name = name
        self.observables = observables
        self.exclude_obs = self.get_exclude_obs_dict()

    def get_exclude_obs_dict(self):
        """Get a dictionary with observables to be excluded from each
        (Fast)Likelihood instance."""
        exclude_obs = {}
        for name, flh in self.likelihood.fast_likelihoods.items():
            exclude_observables = set(flh.observables) - set(self.observables)
            if set(flh.observables) != exclude_observables:
                exclude_obs[name] = exclude_observables
        for name, lh in self.likelihood.likelihoods.items():
            exclude_observables = set(lh.observables) - set(self.observables)
            if set(lh.observables) != exclude_observables:
                exclude_obs[name] = exclude_observables
        return exclude_obs

    def log_likelihood(self, par_dict, wc_obj, delta=False):
         custom_log_likelihood = 0
         for llh_name, exclude_observables in self.exclude_obs.items():
             llh = (self.likelihood.fast_likelihoods.get(llh_name)
                    or self.likelihood.likelihoods.get(llh_name))
             custom_log_likelihood += llh.log_likelihood(
                 self.likelihood.par_dict, wc_obj, delta=delta,
                 exclude_observables=exclude_observables
             )
         return custom_log_likelihood

    def get_number_observations(self):
        """Get the number of observations, defined as individual measurements
        of observables."""
        nobs = 0
        for llh_name, exclude_observables in self.exclude_obs.items():
            if llh_name in self.likelihood.fast_likelihoods:
                flh = self.likelihood.fast_likelihoods[llh_name]
                nobs += len(set(flh.observables) - set(exclude_observables or []))
            else:
                lh = self.likelihood.likelihoods[llh_name]
                ml =  lh.measurement_likelihood
                nobs += ml.get_number_observations(
                    exclude_observables=exclude_observables)
        return nobs


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
        likelihood._check_sm_cov_loaded()
        self.w = w
        self._obstable_tree_cache = None
        self._log_likelihood_dict = None

    def _delta_log_likelihood(self):
        """Compute the delta log likelihood for the individual likelihoods"""
        ll = self.likelihood._log_likelihood(self.w)
        for name in ll:
            ll[name] -= self.likelihood.log_likelihood_sm[name]
        ll['global'] = sum([v for k, v in ll.items() if 'custom_' not in k])
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

    def pvalue_dict(self):
        nobs = self.likelihood.number_observations_dict()
        chi2 = self.chi2_dict()
        return {k: pvalue(chi2[k], nobs[k]) for k in chi2}

    def chi2_dict(self):
        ll = self.log_likelihood_dict()
        llsm = self.likelihood._log_likelihood_sm.copy()
        llsm['global'] = sum([v for k, v in llsm.items() if 'custom_' not in k])
        return {k: -2 * (ll[k] + llsm[k]) for k in ll}

    @property
    def _obstable_tree(self):
        if not self._obstable_tree_cache:
            llh = self.likelihood
            info = copy(llh.obstable_sm)
            for flh_name, flh in llh.fast_likelihoods.items():
                # loop over fast likelihoods: they only have a single "measurement"
                m = flh.pseudo_measurement
                ml = flh.likelihood.measurement_likelihood
                pred = ml.get_predictions_par(llh.par_dict, self.w)
                for i, obs in enumerate(flh.observables):
                    info[obs]['theory'] = pred[obs]
                    ll_central = info[obs]['ll_central']
                    ll_sm = info[obs]['ll_sm']
                    ll = m.get_logprobability_single(obs, pred[obs])
                    # DeltaChi2 is -2*DeltaLogLikelihood
                    info[obs]['pull exp.'] = pull(-2 * (ll - ll_central), dof=1)
                    s = -1 if ll > ll_sm else 1
                    info[obs]['pull SM'] = s * pull(-2 * (ll - ll_sm), dof=1)
            for lh_name, lh in llh.likelihoods.items():
                # loop over "normal" likelihoods
                ml = lh.measurement_likelihood
                pred = ml.get_predictions_par(llh.par_dict, self.w)
                for i, obs in enumerate(lh.observables):
                    info[obs]['theory'] = pred[obs]
                    ll_central = info[obs]['ll_central']
                    ll_sm = info[obs]['ll_sm']
                    p_comb = info[obs]['exp. PDF']
                    ll = p_comb.logpdf([pred[obs]])
                    info[obs]['pull exp.'] = pull(-2 * (ll - ll_central), dof=1)
                    s = -1 if ll > ll_sm else 1
                    info[obs]['pull SM'] = s * pull(-2 * (ll - ll_sm), dof=1)
            self._obstable_tree_cache = info
        return self._obstable_tree_cache

    def obstable(self, min_pull_exp=0, sort_by='pull exp.', ascending=None,
                 min_val=None, max_val=None):
        r"""Return a pandas data frame with the central values and uncertainties
        as well as the pulls with respect to the experimental and the SM values for each observable.

        The pull is defined is $\sqrt(|-2\ln L|)$. Note that the global
        likelihood is *not* simply proportional to the sum of squared pulls
        due to correlations.
        """
        sort_keys = ['name', 'exp. unc.', 'experiment', 'pull SM', 'pull exp.',
                     'th. unc.', 'theory']
        if sort_by not in sort_keys:
            raise ValueError(
                "'{}' is not an allowed value for sort_by. Allowed values are "
                "'{}', and '{}'.".format(sort_by, "', '".join(sort_keys[:-1]),
                                         sort_keys[-1])
            )
        info = self._obstable_tree
        subset = None
        if sort_by == 'pull exp.':
            # if sorted by pull exp., use descending order as default
            if ascending is None:
                ascending = False
            if min_val is not None:
                min_val = max(min_pull_exp, min_val)
            else:
                min_val = min_pull_exp
        elif min_pull_exp != 0:
            subset = lambda row: row['pull exp.'] >= min_pull_exp
        # if sorted not by pull exp., use ascending order as default
        if ascending is None:
            ascending = True
        info = self._obstable_filter_sort(info, sortkey=sort_by,
                                          ascending=ascending,
                                          min_val=min_val, max_val=max_val,
                                          subset=subset)
        # create DataFrame
        df = pd.DataFrame(info).T
        # if df has length 0 (e.g. if min_pull is very large) there are no
        # columns that could be removed
        if len(df) >0:
            # remove columns that are only used internal and should not be
            # included in obstable
            del(df['inspire'])
            del(df['lh_name'])
            del(df['name'])
            del(df['exp. PDF'])
            del(df['ll_central'])
            del(df['ll_sm'])
        return df

    @staticmethod
    def _obstable_filter_sort(info, sortkey='name', ascending=True,
                              min_val=None, max_val=None,
                              subset=None, max_rows=None):
        # impose min_val and max_val
        if min_val is not None:
            info = {obs:row for obs,row in info.items()
                    if row[sortkey] >= min_val}
        if max_val is not None:
            info = {obs:row for obs,row in info.items()
                    if row[sortkey] <= max_val}
        # get only subset:
        if subset is not None:
            info = {obs:row for obs,row in info.items() if subset(row)}
        # sort
        info = OrderedDict(sorted(info.items(), key=lambda x: x[1][sortkey],
                                  reverse=(not ascending)))
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
