import flavio
import numpy as np
from math import sin, cos, sqrt, acos
import inspect
from abc import ABC, abstractmethod


m = flavio.Measurement('CKM ratio measurements')
m.set_constraint('RKpi(P+->munu)', '1.3368+-0.0032')
m.set_constraint('DeltaM_d/DeltaM_s', '0.02852+-0.00011')


def get_ckm_schemes():
    return {
        k: v() for k,v in globals().items()
        if inspect.isclass(v)
        and issubclass(v, CKMScheme)
        and len(v.observables) == 4
    }


class CKMScheme(ABC):
    """Base class for schemes to determine the SMEFT CKM elements from
    a set of four input observables.

    This is an abstract class that contains the following abstract methods to be
    overwritten by its subclasses:
    - ckm_fac: static method with arguments `Vus`, `Vcb`, `Vub`, and
      `delta` that returns a list containing the CKM prefactors of the
      four observables in the order given by the `observales` attribute.
    - get_ckm: static method that is the inverse of the `ckm_fac` method
      such that `get_ckm(ckm_fac(Vus, Vcb, Vub, delta))` will return the
      list `[Vus, Vcb, Vub, delta]`.
    - jacobian: static method with arguments `Vus`, `Vcb`, `Vub`, and
      `delta` that returns the Jacobian of the transformation defined by
      the `ckm_fac` method in terms of a numpy array of shape (4,4).
    Furthermore, subclasses have to overwrite the following class attribute:
    - observables: class attribute containing a list of exactly four
      observable strings. Currently, only observables without arguments
      are supported. They must have existing, uncorrelated experimental
      measurements.
    """

    observables = []

    def __init__(self, par_obj=None):
        """Initialize the class.

        Parameters:
        - par_obj: instance of `flavio.classes.ParameterConstraints`.
          Defaults to `flavio.default_parameters`.
        """
        assert len(self.observables) == 4, "Exactly 4 observables should be specified"
        self.par_obj = par_obj or flavio.default_parameters
        # central parameter values
        self.par_central = self.par_obj.get_central_all()
        # central CKM values
        self.ckm_par = ['Vus', 'Vub', 'Vcb', 'delta']
        self.ckm_initial = {k: self.par_central[k] for k in self.ckm_par}

    @staticmethod
    @abstractmethod
    def ckm_fac(Vus, Vcb, Vub, delta):
        pass

    @staticmethod
    @abstractmethod
    def get_ckm(ckm_fac):
        pass


    @staticmethod
    @abstractmethod
    def jacobian(Vus, Vcb, Vub, delta):
        pass

    def sm_covariance(self, N=1000):
        """Compute the covariance of theory predictions for the four input
        observables in the SM, fixing the CKM elements."""
        par_vary = [p for p in self.par_obj.all_parameters if p not in self.ckm_par]
        return flavio.sm_covariance(self.observables, N=N, par_obj=self.par_obj, par_vary=par_vary)

    def exp_measurements(self):
        """Return a list of four probability distributions, corresponding
        to the  experimental measurements of the four input observables."""
        return {obs: flavio.combine_measurements(obs) for obs in self.observables}

    def exp_covariance(self, measurements):
        """Return the covariance of experimental measurements.
        Correlations are currently neglected!
        `measurements` should be the output returned by the `exp_measurements`
        method."""
        return np.diag([measurements[obs].standard_deviation**2 for obs in self.observables])

    def obs_covariance(self, N=1000):
        """Return the covariance (theory and experiment combined) for
        the four input observables in the SM."""
        sm_cov = self.sm_covariance(N=N)
        measurements = self.exp_measurements()
        exp_cov = self.exp_covariance(measurements)
        return sm_cov + exp_cov

    def np_predictions_nockm(self, w=None, **ckm):
        """Return the predictions for the four input observables in the presence
        of NP (parametrized by a `wilson.Wilson` instance `w`) with the
        CKM prefactor removed."""
        fac = self.ckm_fac(**ckm)
        par_dict = self.par_central.copy()
        par_dict.update(ckm)
        if w is None:
            w = flavio.WilsonCoefficients()
        return [flavio.classes.Observable[obs].prediction_par(par_dict, w) / fac[i]
                for i, obs in enumerate(self.observables)]

    def ckm_covariance(self, N=1000):
        """Return the covariance for the four CKM parameters
        `Vus`, `Vcb`, `Vub`, `delta`"""
        ckm = self.ckm_initial
        J = self.jacobian(**ckm)
        cov = self.obs_covariance(N=N)
        pred_sm = self.np_predictions_nockm(w=None, **ckm)
        cov = cov / np.outer(pred_sm, pred_sm)
        iJ = np.linalg.inv(J)
        return iJ @ cov @ iJ.T

    def ckm_fac_np(self, w, **ckm):
        """Return the central values for the four CKM prefactors
        in the presence of new physics (parametrized by a `wilson.Wilson`
        instance `w`) """
        exp = self.exp_measurements()
        exp_cen = np.array([exp[obs].central_value for obs in self.observables])
        np_cen = np.array(self.np_predictions_nockm(w, **ckm))
        return exp_cen / np_cen

    def _ckm_np(self, w=None, **ckm):
        """Return the central values for the four CKM parameters
        `Vus`, `Vcb`, `Vub`, `delta`,
        in the presence of new physics (parametrized by a `wilson.Wilson`
        instance `w`).

        This is a private method where the CKM paramerters in the `Wilson`
        instance and in `**ckm` must be provided manually. For an iterative
        version where this is done automatically, see `ckm_np`."""
        ckm_fac = self.ckm_fac_np(w, **ckm)
        return self.get_ckm(ckm_fac)

    def ckm_np(self, w=None, iterate=10):
        """Return the central values for the four CKM parameters
        `Vus`, `Vcb`, `Vub`, `delta`,
        in the presence of new physics (parametrized by a `wilson.Wilson`
        instance `w`) """
        Vus, Vcb, Vub, delta = [self.ckm_initial[p] for p in ['Vus', 'Vcb', 'Vub', 'delta']]
        for _ in range(iterate):
            if w is None:
                Vus, Vcb, Vub, delta = self._ckm_np(w=None, Vus=Vus, Vcb=Vcb, Vub=Vub, delta=delta)
            else:
                w.set_option('parameters', {'Vus': Vus, 'Vcb': Vcb, 'Vub': Vub, 'gamma': delta})
                Vus, Vcb, Vub, delta = self._ckm_np(w, Vus=Vus, Vcb=Vcb, Vub=Vub, delta=delta)
        return Vus, Vcb, Vub, delta


class CKMSchemeRmuBtaunuBxlnuDeltaM(CKMScheme):
    """CKM scheme where the four input observables are given by:
    - 'RKpi(P+->munu)' (mostly fixing `Vus`)
    - 'BR(B->Xcenu)' (fixing `Vcb`)
    - 'BR(B+->taunu)' (fixing `Vub`)
    - 'DeltaM_d/DeltaM_s' (mostly fixing `delta`)
    """

    observables=[
        'RKpi(P+->munu)',
        'BR(B->Xcenu)',
        'BR(B+->taunu)',
        'DeltaM_d/DeltaM_s'
    ]

    @staticmethod
    def ckm_fac(Vus, Vcb, Vub, delta):
        """Return the for CKM prefactors as function of the four CKM elements"""
        return [
            (Vus**2 / (1 - Vub**2 - Vus**2)),
            Vcb**2,
            Vub**2,
            -((Vcb**2*Vus**2 + Vub**2*(-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2) -
            2*Vcb*Vub*Vus*((-1 + Vcb**2 + Vub**2)**2*(-1 + Vub**2 + Vus**2)**2)**0.25*cos(delta))/
            (Vcb**2*(-1 + Vub**2) + (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus**2 -
            2*Vcb*Vub*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta)))
        ]

    @staticmethod
    def get_ckm(ckm_fac):
        """Inverse of the `ckm_fac` method: returns the four CKM parameters
        given the four CKM prefactors."""
        f1 = ckm_fac[0]  # Vus**2 / Vud**2
        f2 = ckm_fac[1]  # Vcb**2
        f3 = ckm_fac[2]  # Vub**2
        f4 = ckm_fac[3]  # abs(Vtb * Vts)**2 / abs(Vtb * Vtd)**2
        return [
            sqrt(f1 - f1 * f3) / sqrt(1 + f1),  # Vus
            sqrt(f2),  # Vcb
            sqrt(f3),  # Vub
            # delta
            acos(-((-1 + f3)*(f3 - f3*(f2 + f3) - f2*f4 + f1*(f2 + f3*(-1 + f2 + f3)*f4)))/
            (2.*sqrt(1 + f1)*sqrt(f2)*sqrt(f3)*sqrt(f1 - f1*f3)*
            ((((-1 + f3)**2*(-1 + f2 + f3)**2)/(1 + f1)**2)**0.25 + sqrt(((-1 + f3)*(-1 + f2 + f3))/(1 + f1))*f4)))
        ]

    @staticmethod
    def jacobian(Vus, Vcb, Vub, delta):
        """Return the Jacobian of the transformation from the four CKM
        parameters to the four CKM prefactors."""
        J = np.zeros((4, 4))
        J[0, 0] = (-2*(-1 + Vub**2)*Vus)/(-1 + Vub**2 + Vus**2)**2
        J[0, 2] = (2*Vub*Vus**2)/(-1 + Vub**2 + Vus**2)**2
        J[1, 1] = 2 * Vcb
        J[2, 2] = 2 * Vub
        J[3, 0] = ((2*(-1 + Vub**2)**2*(Vcb**2 + Vub**2)*((Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus*
            sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2)) - Vcb*Vub*(-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + 2*Vus**2)*cos(delta))
            )/(sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*
            (Vcb**2*(-1 + Vub**2) + (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus**2 -
            2*Vcb*Vub*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta))**2))
        J[3, 1] = ((2*Vub*(-1 + Vub**2)**2*(Vcb*Vub*(-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + 2*Vus**2) -
            (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta)))/
            ((-1 + Vcb**2 + Vub**2)*(Vcb**2*(-1 + Vub**2) + (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus**2 -
            2*Vcb*Vub*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta))**2))
        J[3, 2] = ((2*(-1 + Vub**2)*(-(Vub*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*
            (Vcb**2*(-1 + Vub**2)*(-1 + Vcb**2 + 2*Vub**2) +
            (-Vub**4 + Vub**6 + Vcb**4*(3 + Vub**2) + 2*Vcb**2*(-1 + 2*Vub**2 + Vub**4))*Vus**2)) +
            Vcb*Vus*(Vcb**2*(1 - 6*Vub**4 + 5*Vub**6 + (-1 - 2*Vub**2 + 7*Vub**4)*Vus**2) +
            Vub**2*(-1 + Vub**2)*(1 + 3*Vub**4 - Vus**2 + 4*Vub**2*(-1 + Vus**2)) +
            Vcb**4*(-1 + 2*Vub**4 + Vus**2 + Vub**2*(-1 + 3*Vus**2)))*cos(delta)))/
            (sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*
            (Vcb**2*(-1 + Vub**2) + (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus**2 -
            2*Vcb*Vub*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta))**2))
        J[3, 3] = ((2*Vcb*Vub*(-1 + Vub**2)**2*(Vcb**2 + Vub**2)*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*sin(delta))/
            (Vcb**2*(-1 + Vub**2) + (Vcb**2 + (-1 + Vcb**2)*Vub**2 + Vub**4)*Vus**2 -
            2*Vcb*Vub*Vus*sqrt((-1 + Vcb**2 + Vub**2)*(-1 + Vub**2 + Vus**2))*cos(delta))**2)
        return J
