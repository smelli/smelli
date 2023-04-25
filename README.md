[![unit tests](https://github.com/flav-io/flavio/actions/workflows/test.yml/badge.svg)](https://github.com/smelli/smelli/actions/workflows/test.yml)




# smelli – a global likelihood for precision constraints

`smelli` is a Python package providing a global likelihood function in the
space of dimension-six Wilson coefficients in the Standard Model Effective
Field Theory (SMEFT). The likelihood includes contributions from
quark and lepton flavour physics, electroweak precision tests, and other
precision observables.

The package is based on [flavio](https://github.com/flav-io/flavio) for the
calculation of observables and statistical treatment and
[wilson](https://github.com/wilson-eft/wilson) for the running, translation,
and matching of Wilson coefficients.

## Installation

The package requires Python version 3.6 or above. It can be installed with

```bash
python3 -m pip install smelli --user
```


## Documentation

A brief user manual can be found in the paper cited below.

## Citation

If you use `smelli` in a scientific publication, please cite

>  J. Aebischer, J. Kumar, P. Stangl, and D. M. Straub
>
> "A Global Likelihood for Precision Constraints and Flavour Anomalies"
>
>  [arXiv:1810.07698 [hep-ph]](https://arxiv.org/abs/1810.07698)

Please also cite the publications on [flavio](https://arxiv.org/abs/1810.08132) and [wilson](https://arxiv.org/abs/1804.05033), which are the pillars `smelli` is  built on.

## Bugs and feature requests

Please submit bugs and feature requests using
[Github's issue system](https://github.com/smelli/smelli/issues).

## Contributing

The aim of the package is to provide a likelihood in the
space of dimension-6 SMEFT Wilson coefficients using all
relevant available experimental measurements. If you want
to contribute additional observables, the easiest way is
to implement the observable in [flavio](https://github.com/flav-io/flavio). Observables
implemented there can be added to the likelihood simply
by adding a corresponding entry in one of the
[observable YAML files](https://github.com/smelli/smelli/tree/master/smelli/data/yaml).

Alternatively, also observables computed in any other standalone Python package can be incorporated in principle as long as it adheres to the [WCxf standard](https://wcxf.github.io).
If you want to follow this route, please open an [issue](https://github.com/smelli/smelli/issues) to start the discussion on how to integrate it.

## Contributors

Maintainer:

- Peter Stangl (@peterstangl)

Contributors (in alphabetical order):

- Jason Aebischer
- Matěj Hudec
- Matthew Kirk
- Jacky Kumar
- Niladri Sahoo
- Aleks Smolkovič
- Peter Stangl
- David M. Straub

## License

smelli is released under the MIT license.
