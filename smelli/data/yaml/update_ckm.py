import smelli
from smelli.ckm import CKMScheme, get_scheme_definitions
import numpy as np
import yaml

FILE = 'par_ckm'

def save_ckm(scheme_definition,file_path,N=10000):
    scheme = CKMScheme(scheme_definition)
    val = scheme.ckm_np()
    cov = scheme.ckm_covariance(N=10000)
    err = np.sqrt(np.diag(cov))
    corr = cov / np.outer(err, err)

    dat = [
      {
        "parameters": [
          "Vus",
          "Vcb",
          "Vub",
          "delta"
        ],
        "values": {
          "distribution": "multivariate_normal",
          "central_value": list(val),
          "standard_deviation": err.tolist(),
          "correlation": corr.tolist()
        }
      }
    ]

    with open(file_path, 'w') as f:
        yaml.dump(dat, f)

for name, scheme_definition in get_scheme_definitions().items():
    file_path = '{}_{}.yaml'.format(FILE,name)
    save_ckm(scheme_definition,file_path,N=10000)
