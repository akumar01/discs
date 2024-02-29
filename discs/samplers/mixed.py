"""Block gibbs Sampler Class."""

from itertools import product
from discs.common import math_util as math
from discs.samplers import abstractsampler
from discs.samplers.hmc import MaskedHamiltonianMonteCarlo
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import time
import pdb

import tensorflow as tf
import tensorflow_probability as tfp

# Since we need to mask particular degrees of freedom, we need a more customizable
# implementation of HMC. 

# Wrapper around tfp NUTS sampler to fit it into the discs pipeline
class HMCWrapper(abstractsampler.AbstractSampler):

  def __init__(self, config: ml_collections.ConfigDict, model, model_params, use_nuts=False):
    super().__init__(config)
    self.step_size = config.sampler.step_size

    # tfp sampler needs to be instantiated with the model up front

    if use_nuts:
      sampler = tfp.substrates.jax.mcmc.NoUTurnSampler(
              lambda x: model.forward(model_params, x),
              self.step_size,
              max_tree_depth=10,
              max_energy_diff=1000.0,
              unrolled_leapfrog_steps=1,
              parallel_iterations=10,
              experimental_shard_axis_names=None,
              name=None
          )
    else:
      sampler = MaskedHamiltonianMonteCarlo(
                  lambda x: model.forward(model_params, x),
                  self.step_size,
                  num_leapfrog_steps=4)

    self.sampler = sampler

  def step(self, model, rng, x, model_param, state, x_mask=None):
    # Preprocess the state for input to sampler
    internal_state = self.preprocess_state(state)    

    # Update the log probability function (needed to make use of masks)
    self.sampler._impl.inner_kernel._parameters['target_log_prob_fn'] = lambda x: model.forward(model_param, x, x_mask)
    x_new, internal_state = self.sampler.one_step(x, internal_state, rng, mask=x_mask)
    
    # Postprocess the state for input to the sampler
    state = self.postprocess_state(state, internal_state)
    return x_new, state, None

  def make_init_state(self, rng, init_sample):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)

    # create internal sampler state and fold-in to state
    internal_state = self.sampler.bootstrap_results(init_sample)
    state = self.postprocess_state(state, internal_state)
    # Save as an internal attribute
    self.internal_state = internal_state

    return state

  def update_sampler_state(self, sampler_state):

    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + 1) % dim
    sampler_state['num_ll_calls'] += self.num_categories
    return sampler_state

  def preprocess_state(self, sampler_state):

    # Internal state is a named tuple, so need to use
    # ._replace instead of setattr
    replace_dict = {}
    for f in self.internal_state._fields:
      replace_dict[f] = sampler_state[f]      

    self.internal_state = self.internal_state._replace(**replace_dict)
    return self.internal_state

  def postprocess_state(self, state, internal_state):
    for f in internal_state._fields:
        state[f] = getattr(internal_state, f)
    return state

class BlockGibbsSampler(abstractsampler.AbstractSampler):
  """Block gibbs Sampler Class."""

  def __init__(self, dsampler, csampler):
    self.dsampler = dsampler
    self.csampler = csampler

  def make_init_state(self, rng, init_dsample, init_csample):
    """Init sampler state."""

    drng, crng = jax.random.split(rng, 2)
    # Call the methods of the underlying samplers
    dstate = self.dsampler.make_init_state(drng)
    cstate = self.csampler.make_init_state(crng, init_csample)

    state = super().make_init_state(rng)
    state['dstate'] = dstate
    state['cstate'] = cstate

    return state

  def update_sampler_state(self, sampler_state, dstate, cstate):
    sampler_state = super().update_sampler_state(sampler_state)
    sampler_state['dstate'] = dstate
    sampler_state['cstate'] = cstate    
    return sampler_state

  def step(self, model, rng, x_d, x_c, model_param, state, xd_mask=None, xc_mask=None):

    drng, crng = jax.random.split(rng)

    # Step discrete and then continuous
    x_d_new, new_dstate, _ = self.dsampler.step(model, drng, x_d, model_param, state['dstate'], xd_mask)
    x_c_new, new_cstate, _ = self.csampler.step(model, crng, x_c, model_param, state['cstate'], xc_mask)
    new_state = self.update_sampler_state(state, new_dstate, new_cstate)
    return x_d_new, x_c_new, new_state

def build_sampler(config):
  return BlockGibbsSampler(config)
