from tqdm import tqdm
import importlib
import numpy as np
import pdb
import jax
import sys
import jax.numpy as jnp
from ml_collections import config_dict
from discs.models.ising import Ising
from discs.models.configs import ising_config
from discs.models.doublewell import DoubleWell
from discs.samplers.gwg import CategoricalGWGSampler, BinaryGWGSampler

from discs.samplers.dlmc import CategoricalDLMC, BinaryDLMC
from discs.samplers.randomwalk import RWSampler
from discs.samplers.gibbs import GibbsSampler

from discs.common import configs as common_configs
from discs.samplers.configs import dlmc_config, gibbs_config, randomwalk_config, gwg_config

def calc_hamming_dist_series(samples):
    s0 = samples[0]
    dist = []
    for i in range(len(samples[1:])):
        dist.append(jnp.sum(samples[i] - s0))
    return jnp.array(dist)

if __name__ == '__main__':

    burnin = 1000
    n_chains = 1
    chain_length = 50000

    # Double Well Configuration
    # model_class = DoubleWell
    pm=2
    dim=10
    model_config = dict(
        dim=dim,
        range=pm,
        shape=(dim,),
        num_categories=int(2*pm+1),
        temp=1,
        offset=40,
        name='double_well'
    )

    # Ising configuration
    model_class = Ising
    model_config = ising_config.get_config()

    config = common_configs.get_config()
    model_config = config_dict.ConfigDict(model_config)
    config.model.update(model_config)

    # Samplers to test
    #samplers = ['dlmc', 'gwg', 'randomwalk', 'gibbs']
    samplers = ['dlmc']
    sampler_dict = {'dlmc':BinaryDLMC, 
                    'gwg': BinaryGWGSampler,
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
            step_rng_, step_rng = jax.random.split(step_rng, 2)
            x, sampler_state, _ = sampler.step(model, step_rng_, x, model_param, sampler_state)
            if idx > burnin:
                samples.append(x)

    hdist = calc_hamming_dist_series(samples)
    pdb.set_trace()