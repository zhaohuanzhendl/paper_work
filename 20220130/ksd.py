### imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class RBFKernel:
    """radial basis (gaussian) kernel"""
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def eval(self, x, y):
        dist = torch.sum((x-y)**2)
        return torch.exp(- dist / (2 * self.sigma**2))
    
# $$ \mathcal{A}_p\phi(x) = \phi(x)\nabla_x\log{p(x)}^T + \nabla_x\phi(x) $$
### stein operator
def stein_op(p, phi, x):
    """
    implementation of the Langevin-Stein operator, which is an operator
    variational objective useful for SVGD.
    """
    # attractive term
    x_a = x.detach().requires_grad_()
    #print("x_a:",x_a)
    log_prob = torch.log(p(x_a))
    log_prob.backward()
    a_term = torch.dot(phi(x_a).view(-1), x_a.grad)
    
    # repulsive term
    r_term = 0
    try:
        x_r = x.detach().requires_grad_()
        #print("yes x_r:", x_r)
        phi_x = phi(x_r)
        #print("yes phi_x:", phi_x)
        phi_x.backward(torch.ones_like(x_r))
        r_term = torch.sum(x_r.grad)
        #print("yes r_term:", r_term)
    except:
        print("err x_r:", x_r)
        print("err phi_x:", phi_x)
    return a_term + r_term

def p(x):
    return torch.exp(x)

def phi(x):
    """
    torch tensor version of N(0,1) gaussian
    """
    res_phi = torch.exp(torch.distributions.Normal(0,1).log_prob(x))
    #print("res_phi:", res_phi)
    return res_phi

#####################################################################
#Stein discrepancy
#####################################################################
def ls_expectation(p, q, phi, num_samples):
    """
    compute monte carlo estimate of the (nonkernelized)
    stein discrepancy, where samples come from q.
    """
    x = q.sample((num_samples,))
    ls_vals = torch.zeros(x.shape)
    for i in range(len(x)):
        #if i<10:
        #    print("x[i].view(-1):",x[i].view(-1))
        ls_vals[i] = stein_op(p, phi, x[i].view(-1))
    
    return torch.mean(ls_vals)

q = torch.distributions.Normal(0,1)
lss = []
#for i in tqdm(range(1, 150)):
for i in tqdm(range(1, 15)):
    num_samples = 10 * i
    #print(p,q,phi)
    ls = ls_expectation(p, q, phi, num_samples).item()
    lss.append(ls)
    
plt.plot(lss)
plt.show()

#####################################################################
#Kernel Stein discrepancy
#Here, $\phi^*_{q,p}(-) = \mathbb{E}_{x\sim q}[\mathcal{A}_p k(x,-)]$, and in this case,
#
#$$ \mathbf{S}(q,p) = \mathbb{E}_{x\sim p}[\operatorname{trace}{(\mathcal{A}_p\phi^*_{q,p}(x))}^2]  $$
#####################################################################
class KSD:
    """
    Kernelized Stein discrepancy for a kernel in a
    given RKHS.
    """
    def __init__(self, kernel, p):
        self.kernel = kernel
        self.p = p
        
    def optimal_fn(self, q, samples):
 
        def kern_curry(y):
            def k(x):
                return self.kernel.eval(x,y).view(-1)
            return k
        
        def phi(x):
            num_samples = samples.shape[0]
            phi_vals = torch.zeros(samples.shape)
            for i in range(num_samples):
                sp = samples[i]
                phi_vals[i] = stein_op(self.p, kern_curry(sp), x.view(-1))
            #print("torch.mean(phi_vals):", torch.mean(phi_vals).view(-1))
            return torch.mean(phi_vals).view(-1)
    
        return phi
            
    def eval(self, q, num_samples=10):
        """
        Monte carlo estimate of KSD.
        """
        samples = q.sample((num_samples,))
        phi = self.optimal_fn(q, samples)
        # get new samples for second expectation
        new_samples = q.sample((num_samples,))
        ksd_vals = torch.zeros(samples.shape)
        ls_ksd = []

        for i in tqdm(range(num_samples)):
            sp = new_samples[i]
            #print(self.p)
            #print(phi)
            #print(sp.view(-1))
            #break
            ksd_vals[i] = stein_op(self.p, phi, sp.view(-1))
            ls_ksd.append(ksd_vals[i].detach().numpy())
        plt.plot(ls_ksd)
        plt.show()
        return torch.mean(ksd_vals)
    
ksd = KSD(rbf_kernel, p)
q = torch.distributions.Normal(0,1)
result = ksd.eval(q, num_samples=100)
print(['#']*10)
print("result:",result)
