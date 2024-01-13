from tqdm import tqdm
import importlib
import numpy as np
import pdb
import jax
import sys
import jax.numpy as jnp
from ml_collections import config_dict
from discs.models.ergm import ERGM
from discs.models.configs import ergm_config
from discs.models.doublewell import DoubleWell
from discs.samplers.gwg import CategoricalGWGSampler

from discs.samplers.dlmc import CategoricalDLMC
from discs.samplers.randomwalk import RWSampler
from discs.samplers.gibbs import GibbsSampler

from discs.common import configs as common_configs
from discs.samplers.configs import dlmc_config, gibbs_config, randomwalk_config, gwg_config

from discs.evaluators.ess_eval import ESSevaluator

from copy import deepcopy


burnin = 1000  #10000
n_chains = 1
chain_length = 10000 #int(1e5) 


# ERGM configuration
model_class = ERGM
model_config = ergm_config.get_config()

config = common_configs.get_config()
model_config = config_dict.ConfigDict(model_config)
config.model.update(model_config)

# Samplers to test
#samplers = ['dlmc', 'gwg', 'randomwalk', 'gibbs']
samplers = ['gibbs']
sampler_dict = {'dlmc':CategoricalDLMC, 
                'gwg': CategoricalGWGSampler,
                'randomwalk':RWSampler,
                'gibbs':GibbsSampler}
sampler_config_dict = {'dlmc':dlmc_config, 'gwg':gwg_config, 'randomwalk':randomwalk_config, 'gibbs':gibbs_config}

# Initialize each chain at the same seed
rnd = jax.random.PRNGKey(165)
init_rng, step_rng, model_rng = jax.random.split(rnd, 3)

ess = {}
for s in tqdm(samplers):

    # Load sampler configuration
    sampler_config = sampler_config_dict[s].get_config()
    sample_init_rng, sampler_init_rng = jax.random.split(init_rng, 2)
    config.sampler.update(sampler_config)
    sampler = sampler_dict[s](config)
    # Initialize
    model = model_class(model_config)
    model_param = model.make_init_params(model_rng)
    x = model.get_init_samples(sample_init_rng, n_chains)
    sampler_state = sampler.make_init_state(sampler_init_rng)
    samples = []
    for idx in tqdm(range(burnin + chain_length)):
        #x_old = deepcopy(x)
        x, sampler_state, _ = sampler.step(model, step_rng, x, model_param, sampler_state)
        if idx > burnin:
            samples.append(x)
            #print(x-x_old)
            
    np.save("samples.npy", samples)

    
# Calculate effective sample size
rnd_ess = model_rng
esseval = ESSevaluator(config)
ess,std = esseval._get_ess(rnd_ess, jnp.array(samples))
print(ess,std)