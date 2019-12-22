import autograd.numpy as np
import matplotlib.pyplot as plt
from normflows import distributions


mus = np.array([[0], [4]])
Sigma_diags = np.array([[1], [1]])
# probs = np.array([0.3, 0.7])
pi = np.array([0.3])
xx = np.linspace(-3, 8, 100).reshape(-1, 1)
yy = distributions.prob_gm(xx, mus, Sigma_diags, pi)
plt.plot(xx, yy)
plt.show()
