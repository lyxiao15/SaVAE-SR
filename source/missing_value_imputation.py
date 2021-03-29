import torch
from torch.autograd import Variable
from .network import VAE

def mcmc_missing_imputation(observe_normal, vae, n_iteration, x):
    if isinstance(x, Variable):
        test_x = Variable(x.data)
    else:
        test_x = Variable(x)

    with torch.no_grad():
        for mcmc_step in range(n_iteration):
            p_xz, _, _ = vae(x)
            test_x[observe_normal == 0.] = p_xz.sample()[0][observe_normal == 0.]
    return test_x

