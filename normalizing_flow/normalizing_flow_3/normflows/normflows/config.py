from pathlib import Path

import autograd.numpy.random as npr


root = Path(__file__).parent.parent.parent
data = root / 'data'
mnist = data / 'mnist'
notebooks = root / 'notebooks'
figs = root / 'figures'
models = root / 'models'
results = root / 'results'

figname = str(figs / '{}_flows_iter_{}.png')
rs = npr.RandomState(101)
