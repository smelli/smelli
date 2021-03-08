import unittest
from .classes import *


class TestGlobalLikelihood(unittest.TestCase):

    def test_init(self):
        gl = GlobalLikelihood()
        # check that the SM delta ll is 0
        ll_sm = gl.parameter_point({}, 100)
        for k, v in ll_sm.log_likelihood_dict().items():
            self.assertEqual(v, 0, msg="Failed for {}".format(k))
        # check just that this does not give an error
        ll_np = gl.parameter_point({'lq1_2223': 1e-8}, 1000)

    def test_dispatch(self):
        wc_dict  = {'phiD': 1e-10, 'lq1_3323': 1e-7}
        gl = GlobalLikelihood()
        pp = []
        pp.append(gl.parameter_point(wc_dict, 1e3))  # dict, int
        pp.append(gl.parameter_point(wc_dict, scale=1e3))  # dict, kwarg
        pp.append(gl.parameter_point(wc_dict, 1.e3))  # dict, float
        w = Wilson(wc_dict, 1e3, 'SMEFT', 'Warsaw')
        pp.append(gl.parameter_point(w))  # dict, float
        filename = get_datapath('smelli', 'data/test/wcxf.yaml')
        pp.append(gl.parameter_point(filename))  #str
        for i, p in enumerate(pp):
            self.assertDictEqual(wc_dict, p.w.wc.dict,
                                 msg="Failed for {}".format(i))
            self.assertEqual(1e3, p.w.wc.scale,
                             msg="Failed for {}".format(i))

    def test_basis(self):
        gl = GlobalLikelihood(basis='Warsaw up')
        self.assertEqual(gl.basis, 'Warsaw up')

    def test_wet(self):
        gl = GlobalLikelihood(eft='WET')
        # check that the SM delta ll is 0
        ll_sm = gl.parameter_point({}, 100)
        for k, v in ll_sm.log_likelihood_dict().items():
            self.assertEqual(v, 0, msg="Failed for {}".format(k))
        # check just that this does not give an error
        ll_np = gl.parameter_point({'C9_bsmumu': -1}, 4.8)

    def test_incl_excl(self):
        gl = GlobalLikelihood(eft='WET', basis='flavio', include_likelihoods=['fast_likelihood_quarks.yaml'])
        ll = gl.parameter_point({}, 100).log_likelihood_dict()
        self.assertSetEqual(set(ll.keys()), {'fast_likelihood_quarks.yaml', 'global'})
        gl = GlobalLikelihood(eft='WET', basis='flavio', include_likelihoods=['likelihood_lfv.yaml'])
        ll = gl.parameter_point({}, 100).log_likelihood_dict()
        self.assertSetEqual(set(ll.keys()), {'likelihood_lfv.yaml', 'global'})
        gl = GlobalLikelihood(eft='WET', basis='flavio', exclude_likelihoods=['likelihood_lfv.yaml'])
        ll = gl.parameter_point({}, 100).log_likelihood_dict()
        self.assertNotIn('likelihood_lfv.yaml', set(ll.keys()))
        self.assertRaises(ValueError, GlobalLikelihood, include_likelihoods=["nonexistent_likelihood.yaml"])
        self.assertRaises(ValueError, GlobalLikelihood, exclude_likelihoods=["nonexistent_likelihood.yaml"])

    def test_chi2_min(self):
        gl_ewpt = GlobalLikelihood(fix_ckm=True, include_likelihoods=[
            'likelihood_ewpt.yaml',
        ])
        def wc_fct_2D_2args(S,T):
            return {
                'phiWB': S * 7.643950529889027e-08,
                'phiD': -T * 2.5793722852276787e-07,
            }
        def wc_fct_2D_1arg(x):
            S,T = x
            return {
                'phiWB': S * 7.643950529889027e-08,
                'phiD': -T * 2.5793722852276787e-07,
            }
        def wc_fct_1D(S):
            return {
                'phiWB': S * 7.643950529889027e-08,
            }
        min_data_2D_2args = gl_ewpt.chi2_min(
            wc_fct_2D_2args,
            91.1876
        )
        min_data_2D_1arg = gl_ewpt.chi2_min(
            wc_fct_2D_1arg,
            91.1876,
            n=2
        )
        min_data_1D = gl_ewpt.chi2_min(
            wc_fct_1D,
            91.1876,
            n=1
        )
        self.assertEqual(len(min_data_1D['global']['coords_min']), 1)
        self.assertEqual(len(min_data_2D_2args['global']['coords_min']), 2)
        self.assertEqual(
            len(min_data_1D['likelihood_ewpt.yaml']['coords_min']),
            1
        )
        self.assertEqual(
            len(min_data_2D_2args['likelihood_ewpt.yaml']['coords_min']),
            2
        )
        self.assertEqual(
            min_data_2D_2args['global']['z_min'],
            min_data_2D_1arg['global']['z_min'],
        )
        self.assertTrue(all(
            min_data_2D_2args['global']['coords_min']
            ==
            min_data_2D_1arg['global']['coords_min']
        ))
        self.assertEqual(
            min_data_2D_2args['likelihood_ewpt.yaml']['z_min'],
            min_data_2D_1arg['likelihood_ewpt.yaml']['z_min'],
        )
        self.assertTrue(all(
            min_data_2D_2args['likelihood_ewpt.yaml']['coords_min']
            ==
            min_data_2D_1arg['likelihood_ewpt.yaml']['coords_min']
        ))
        with self.assertWarns(UserWarning):
            gl_ewpt.chi2_min(
                wc_fct_2D_2args,
                91.1876,
                n=2
            )
        with self.assertRaises(ValueError):
            gl_ewpt.chi2_min(
                wc_fct_1D,
                91.1876,
            )
        with self.assertRaises(ValueError):
            gl_ewpt.chi2_min(
                wc_fct_2D_1arg,
                91.1876,
            )
        # define a global function for multiprocessing
        global _wc_fct_2D_2args
        def _wc_fct_2D_2args(S,T):
            return {
                'phiWB': S * 7.643950529889027e-08,
                'phiD': -T * 2.5793722852276787e-07,
            }
        min_data_2D_2threads = gl_ewpt.chi2_min(
            _wc_fct_2D_2args,
            91.1876,
            threads=2,
        )
        self.assertEqual(len(min_data_2D_2threads['global']['coords_min']), 2)
        self.assertEqual(
            len(min_data_2D_2threads['likelihood_ewpt.yaml']['coords_min']),
            2
        )
        self.assertEqual(
            min_data_2D_2threads['global']['z_min'],
            min_data_2D_2args['global']['z_min'],
        )
        self.assertEqual(
            min_data_2D_2threads['likelihood_ewpt.yaml']['z_min'],
            min_data_2D_2args['likelihood_ewpt.yaml']['z_min'],
        )
        self.assertTrue(all(
            min_data_2D_2threads['global']['coords_min']
            ==
            min_data_2D_2args['global']['coords_min']
        ))
        self.assertTrue(all(
            min_data_2D_2threads['likelihood_ewpt.yaml']['coords_min']
            ==
            min_data_2D_2args['likelihood_ewpt.yaml']['coords_min']
        ))


class TestGlobalLikelihoodPoint(unittest.TestCase):

    def test_obstable(self):
        gl = GlobalLikelihood()
        res = gl.parameter_point({'lq1_2223': 1e-8}, 91.1876)
        self.assertIsInstance(res, GlobalLikelihoodPoint)
        df = res.obstable(min_pull_exp=1)
        self.assertTrue(df['pull exp.'].min() >= 1)

    def test_pvalue(self):
        gl = GlobalLikelihood()
        res = gl.parameter_point({}, 91.1876)
        pval = res.pvalue_dict()
        self.assertTrue(0 < pval['global'] < 1)
