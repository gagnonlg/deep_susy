import time

import numpy as np
import ROOT

from deep_susy import evaluation


s = np.random.uniform(1, 100, size=1000)
b = np.random.uniform(0, 10, size=1000) * s
u = np.random.uniform(0, 1, size=1000)

# just to load RooStats
ROOT.RooStats.NumberCountingUtils.BinomialExpZ(1., 1., 1.)

tr0 = time.time()
r_roostats = np.zeros((1000,))
for i in range(1000):
    r_roostats[i] = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s[i], b[i], u[i])
dtr = time.time() - tr0

td0 = time.time()
r_ds = evaluation.significance(s, b, u)
dtd = time.time() - td0

np.testing.assert_allclose(r_roostats, r_ds)

print 'test successful :)'
print 'roostats: %.3fs' % dtr
print 'deep_susy: %.3fs' % dtd
