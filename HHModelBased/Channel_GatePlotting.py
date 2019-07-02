import numpy as np
import math
import matplotlib.pyplot as plt
def KA_n_gate():
    t = np.linspace(-100, 100, int(200/0.01))
    alpha_out = []
    for item in t:
        alpha_out.append(KA_n_alpha(item))
    plt.plot(t, alpha_out)
    plt.show()

def KA_n_alpha(x):
    alpha = 0.02 * math.exp(1.8 * (x + 33.6) * 38.2997 * 0.001)
    return alpha

KA_n_gate()