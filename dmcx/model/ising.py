"""Ising Energy Function."""

import os
from os.path import exists

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
from dmcx.model import abstractmodel
from tqdm import tqdm
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import pdb
import jax
import jax.numpy as jnp
import ml_collections
import pickle


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):

    if isinstance(config.shape, int):
      self.shape = (config.shape, config.shape)
    else:
      self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.external_field_type = config.external_field_type
    self.init_sigma = config.init_sigma
    path = os.getcwd()[:os.getcwd().find("discrete_mcmc"
                                        )] + "discrete_mcmc/dmcx/model/"
    self.file_to_mean_and_var = path + "mean_var_2D_ising_with_shape_{}_lambda_{}_forcetype_{}.pkl".format(
        self.shape[0], self.lambdaa, self.external_field_type)
    if not exists(self.file_to_mean_and_var):
      self.expected_val = None
      self.var = None
    else:
      with open(self.file_to_mean_and_var, "rb") as f:
        data = pickle.load(f)
        self.expected_val = data["mean"]
        self.var = data["var"]
    self.setup_sampler(config)

  def setup_sampler(self, config: ml_collections.ConfigDict):

    self.sampler_config = ml_collections.ConfigDict(
        initial_dictionary=dict(
            sample_shape=self.shape,
            num_categories=2,
            random_order=False,
            block_size=3))
    self.sampler = blockgibbs_sampler.BlockGibbsSampler(self.sampler_config)
    self.parallel_sampling = config.parallel_sampling
    self.sampler_convergance_threshold = config.sampler_convergance_threshold

  def make_init_params(self, rnd):
    # connectivity strength
    params_weight_h = self.lambdaa * jnp.ones(self.shape)
    params_weight_v = self.lambdaa * jnp.ones(self.shape)

    # external force
    if self.external_field_type == 1:
      params_b = jax.random.normal(rnd, shape=self.shape) * self.init_sigma
    else:
      params_b = jnp.zeros(shape=self.shape)
    return jnp.array([params_b, params_weight_h, params_weight_v])

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=2,
        dtype=jnp.int32)
    return x0

  def forward(self, params, x):

    x = 2 * x - 1
    w_b = params[0]
    w_h = params[1][:, :-1]
    w_v = params[2][:-1, :]
    sum_neighbors = jnp.zeros((x.shape[0],) + self.shape)
    sum_neighbors = sum_neighbors.at[:, :, :-1].set(
        sum_neighbors[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h)  # right
    sum_neighbors = sum_neighbors.at[:, :, 1:].set(
        sum_neighbors[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h)  # left
    sum_neighbors = sum_neighbors.at[:, :-1, :].set(
        sum_neighbors[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v)  # down
    sum_neighbors = sum_neighbors.at[:, 1:, :].set(
        sum_neighbors[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v)  # up
    biases = w_b * x
    loglikelihood = sum_neighbors + biases
    loglikelihood = jnp.sum((loglikelihood).reshape(x.shape[0], -1), axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  def get_expected_val(self, params):
    if self.expected_val is not None:
      return self.expected_val
    return self.compute_mean_and_var(params)[0]

  def get_var(self, params):
    if self.var is not None:
      return self.var
    return self.compute_mean_and_var(params)[1]

  def compute_mean_and_var(self, params):

    rnd = jax.random.PRNGKey(0)
    samples = self.generate_chain_of_samples(rnd, params)
    self.expected_val = self.get_expected_val_from_samples(samples)
    self.var = self.get_var_from_samples(samples)
    ising_model = {}
    ising_model["mean"] = self.expected_val
    ising_model["var"] = self.var
    f = open(self.file_to_mean_and_var, "wb")
    pickle.dump(ising_model, f)
    f.close()
    return [self.expected_val, self.var]

  def generate_chain_of_samples(self, rnd, params):
    """Using Block Gibbs Sampler Generates Chain of Samples."""

    num_samples = 100
    chain_length = 1000
    rng_x0, rng_sampler, rng_sampler_step = jax.random.split(rnd, num=3)
    del rnd
    state = self.sampler.make_init_state(rng_sampler)
    x = self.get_init_samples(rng_x0, num_samples)
    if self.parallel_sampling:
      sampler_step_fn = jax.pmap(
          self.sampler.step, static_broadcasted_argnums=[0])
      n_devices = jax.local_device_count()
      params = jnp.stack([params] * n_devices)
      state = jnp.stack([state] * n_devices)
      x = self.split(x, n_devices)
      rnd_split_num = n_devices
    else:
      sampler_step_fn = jax.jit(self.sampler.step, static_argnums=0)
      rnd_split_num = 2

    samples = self.compute_chains(rng_sampler_step, chain_length, num_samples,
                                  sampler_step_fn, params, state, x,
                                  rnd_split_num)

    if self.parallel_sampling:
      samples = samples.reshape((num_samples,) + self.shape)

    return samples

  def compute_chains(self, rng_sampler_step, chain_length, num_samples,
                     sampler_step_fn, params, state, x, rnd_split_num):
    samples = None
    for i in tqdm(range(chain_length)):
      rng_sampler_step_p = jax.random.split(rng_sampler_step, num=rnd_split_num)
      x, state = sampler_step_fn(self, rng_sampler_step_p, x, params, state)
      del rng_sampler_step_p
      rng_sampler_step, _ = jax.random.split(rng_sampler_step)
      if ((i + 1) % 5 == 0 and self.samples_converged(samples_prev, samples)):
        break
      samples_prev = samples
      samples = x

    return samples

  def samples_converged(self, samples1, samples2):

    if self.parallel_sampling:
      samples1 = samples1.reshape((samples1.shape[0] * samples1.shape[1],) +
                                  self.shape)
      samples2 = samples1.reshape((samples2.shape[0] * samples2.shape[1],) +
                                  self.shape)
    mean1 = self.get_expected_val_from_samples(samples1)
    mean2 = self.get_expected_val_from_samples(samples2)
    if jnp.mean((mean1 - mean2)**2) < self.sampler_convergance_threshold:
      return True
    return False

  def get_expected_val_from_samples(self, samples):
    """Computes distribution expected value from samples."""
    mean = jnp.mean(samples, axis=0)
    return mean

  def get_var_from_samples(self, samples):
    """Computes distribution variance from samples."""
    sample_mean = jnp.mean(samples, axis=0, keepdims=True)
    var = jnp.sum((samples - sample_mean)**2, axis=0) / (samples.shape[0]-1)
    return var

  def split(self, arr, n_devices):
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])
