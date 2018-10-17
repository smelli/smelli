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
        pp.append(gl.parameter_point(wc_dict, 1.e3))  # dict, float
        w = Wilson(wc_dict, 1e3, 'SMEFT', 'Warsaw')
        pp.append(gl.parameter_point(w))  # dict, float
        filename = get_datapath('smelli', 'data/test/wcxf.yaml')
        pp.append(gl.parameter_point(filename))  #str
        for i, p in enumerate(pp):
            self.assertDictEqual(wc_dict, p.w.wc.dict,
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


class TestGlobalLikelihoodPoint(unittest.TestCase):

    def test_obstable(self):
        gl = GlobalLikelihood()
        res = gl.parameter_point({'lq1_2223': 1e-8}, 91.1876)
        self.assertIsInstance(res, GlobalLikelihoodPoint)
        df = res.obstable(min_pull=1)
        self.assertTrue(df['pull'].min() >= 1)
