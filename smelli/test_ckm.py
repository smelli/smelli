from . import ckm
import unittest
import flavio
from wilson import Wilson
from math import sqrt
import smelli
from cmath import exp


par = flavio.default_parameters.get_central_all()


class TestCKM(unittest.TestCase):
    def test_sm(self):
        scheme = ckm.CKMSchemeRmuBtaunuBxlnuDeltaM()
        scheme.ckm_covariance()
        Vus, Vcb, Vub, delta = scheme.ckm_np(None)
        self.assertAlmostEqual(Vus, 0.225, delta=0.001)
        self.assertAlmostEqual(Vcb, 0.041, delta=0.0015)
        self.assertAlmostEqual(Vub, 0.004, delta=0.0005)
        self.assertAlmostEqual(delta, 1.15, delta=0.2)

    def test_ckm_np(self):
        scheme = ckm.CKMSchemeRmuBtaunuBxlnuDeltaM()
        w = Wilson({'lq3_1123': 0.04e-5,
                    'lq3_3313': 0.008e-5,
                    'lq3_2212': 0.5e-7,
                    'qq1_2323': 1e-12,
                    }, 91.1876, 'SMEFT', 'Warsaw')
        Vus, Vcb, Vub, delta = scheme.ckm_np(w, iterate=10)
        w.set_option('parameters', {'Vus': Vus, 'Vcb': Vcb, 'Vub': Vub, 'gamma': delta})
        _Vus, _Vcb, _Vub, _delta = scheme._ckm_np(w, Vus=Vus, Vcb=Vcb, Vub=Vub, delta=delta)
        self.assertAlmostEqual(_Vus, Vus, places=2)
        self.assertAlmostEqual(_Vcb, Vcb, places=2)
        self.assertAlmostEqual(_Vub, Vub, places=2)
        self.assertAlmostEqual(_delta, delta, places=2)


class TestSmelliCKM(unittest.TestCase):
    def test_init(self):
        gl = smelli.GlobalLikelihood()
        # with fix_ckm
        gl_fixckm = smelli.GlobalLikelihood(fix_ckm=True)
        self.assertEqual(gl.par_dict_default['Vcb'], par['Vcb'])
        VcbSM = gl.par_dict_sm['Vcb']
        VubSM = gl.par_dict_sm['Vub']
        VusSM = gl.par_dict_sm['Vus']
        deltaSM = gl.par_dict_sm['delta']
        self.assertAlmostEqual(par['Vcb'], VcbSM, delta=0.0002)
        self.assertAlmostEqual(par['Vub'], VubSM, delta=0.0005)
        self.assertAlmostEqual(par['Vus'], VusSM, delta=0.0006)
        pre = -4 * par['GF'] / sqrt(2)
        # Vcb
        w = Wilson({'lq3_1123': 0.5 * pre * VcbSM * (-0.5)}, 91.1876, 'SMEFT', 'Warsaw')
        pp = gl.parameter_point(w)
        self.assertAlmostEqual(pp.par_dict_np['Vcb'] / VcbSM,  1.5, delta=0.03)
        # with fix_ckm
        pp = gl_fixckm.parameter_point(w)
        self.assertEqual(pp.par_dict_np['Vcb'] / par['Vcb'],  1)
        # Vub
        w = Wilson({'lq3_3313': 0.5 * pre * VubSM * (-0.5) * exp(-1j * deltaSM)}, 91.1876, 'SMEFT', 'Warsaw')
        pp = gl.parameter_point(w)
        self.assertAlmostEqual(pp.par_dict_np['Vub'] / VubSM, 1.5, delta=0.03)
        # Vus
        w = Wilson({'lq3_2212': 0.5 * pre * VusSM * (-0.1)}, 91.1876, 'SMEFT', 'Warsaw')
        pp = gl.parameter_point(w)
        self.assertAlmostEqual(pp.par_dict_np['Vus'] / VusSM, 1.1, delta=0.03)

    def test_fast_likelihoods(self):
        scheme = ckm.CKMSchemeRmuBtaunuBxlnuDeltaM()
        ckm_central = scheme.ckm_np()
        gl = smelli.GlobalLikelihood()
        for fl in gl.fast_likelihoods.values():
            par = fl.par_obj
            self.assertAlmostEqual(par.get_central('Vus'), ckm_central[0], delta=0.00001)
            self.assertAlmostEqual(par.get_central('Vcb'), ckm_central[1], delta=0.00001)
            self.assertAlmostEqual(par.get_central('Vub'), ckm_central[2], delta=0.00001)
            self.assertAlmostEqual(par.get_central('delta'), ckm_central[3], delta=0.0001)
