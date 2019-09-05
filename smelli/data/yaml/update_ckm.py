import smelli
import numpy as np
import yaml

FILE = 'par_ckm.yaml'


scheme = smelli.ckm.CKMSchemeRmuBtaunuBxlnuDeltaM()
val = scheme.ckm_np()
cov = scheme.ckm_covariance()
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


with open(FILE, 'w') as f:
    yaml.dump(dat, f)
