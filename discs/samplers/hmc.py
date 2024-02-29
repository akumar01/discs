# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Hamiltonian Monte Carlo, a gradient-based MCMC algorithm."""
import pdb
import jax
import warnings
import jax.numpy as jnp
from tensorflow_probability.python.internal.backend.jax.compat import v1 as tf1
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf

from tensorflow_probability.substrates.jax.internal import distribute_lib
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.internal import prefer_static as ps
from tensorflow_probability.substrates.jax.internal import samplers
from tensorflow_probability.substrates.jax.mcmc import metropolis_hastings
from tensorflow_probability.substrates.jax.mcmc.internal import leapfrog_integrator as leapfrog_impl
from tensorflow_probability.substrates.jax.mcmc.internal import util as mcmc_util

from tensorflow_probability.substrates.jax.mcmc import (HamiltonianMonteCarlo, 
                                                        UncalibratedHamiltonianMonteCarlo)

# Need to copy over MetropolisHastings as well to propagate the mask argument...
class MaskedMetropolisHastings(metropolis_hastings.MetropolisHastings):
  def one_step(self, current_state, previous_kernel_results, seed=None, mask=None):
    is_seeded = seed is not None
    seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
    proposal_seed, acceptance_seed = samplers.split_seed(seed)

    with tf.name_scope(mcmc_util.make_name(self.name, 'mh', 'one_step')):
      # Take one inner step.
      inner_kwargs = dict(seed=proposal_seed) if is_seeded else {}
      [
          proposed_state,
          proposed_results,
      ] = self.inner_kernel.one_step(
          current_state,
          previous_kernel_results.accepted_results,
          mask=mask,
          **inner_kwargs)
      if mcmc_util.is_list_like(current_state):
        proposed_state = tf.nest.pack_sequence_as(current_state, proposed_state)

      if (not metropolis_hastings.has_target_log_prob(proposed_results) or
          not metropolis_hastings.has_target_log_prob(previous_kernel_results.accepted_results)):
        raise ValueError('"target_log_prob" must be a member of '
                         '`inner_kernel` results.')

      # Compute log(acceptance_ratio).
      to_sum = [proposed_results.target_log_prob,
                -previous_kernel_results.accepted_results.target_log_prob]
      try:
        if (not mcmc_util.is_list_like(
            proposed_results.log_acceptance_correction)
            or proposed_results.log_acceptance_correction):
          to_sum.append(proposed_results.log_acceptance_correction)
      except AttributeError:
        warnings.warn('Supplied inner `TransitionKernel` does not have a '
                      '`log_acceptance_correction`. Assuming its value is `0.`')
      log_accept_ratio = mcmc_util.safe_sum(
          to_sum, name='compute_log_accept_ratio')

      # If proposed state reduces likelihood: randomly accept.
      # If proposed state increases likelihood: always accept.
      # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
      #       ==> log(u) < log_accept_ratio
      log_uniform = tf.math.log(
          samplers.uniform(
              shape=ps.shape(proposed_results.target_log_prob),
              dtype=dtype_util.base_dtype(
                  proposed_results.target_log_prob.dtype),
              seed=acceptance_seed))
      is_accepted = log_uniform < log_accept_ratio

      next_state = mcmc_util.choose(
          is_accepted,
          proposed_state,
          current_state,
          name='choose_next_state')

      kernel_results = metropolis_hastings.MetropolisHastingsKernelResults(
          accepted_results=mcmc_util.choose(
              is_accepted,
              # We strip seeds when populating `accepted_results` because unlike
              # other kernel result fields, seeds are not a per-chain value.
              # Thus it is impossible to choose between a previously accepted
              # seed value and a proposed seed, since said choice would need to
              # be made on a per-chain basis.
              mcmc_util.strip_seeds(proposed_results),
              previous_kernel_results.accepted_results,
              name='choose_inner_results'),
          is_accepted=is_accepted,
          log_accept_ratio=log_accept_ratio,
          proposed_state=proposed_state,
          proposed_results=proposed_results,
          extra=[],
          seed=seed,
      )

      return next_state, kernel_results


class MaskedHamiltonianMonteCarlo(HamiltonianMonteCarlo):

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):

    self._impl = MaskedMetropolisHastings(
        inner_kernel=MaskedUncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
        )).experimental_with_shard_axes(experimental_shard_axis_names)
    self._parameters = self._impl.inner_kernel.parameters.copy()

  def one_step(self, current_state, previous_kernel_results, seed=None, mask=None):
    """Runs one iteration of Hamiltonian Monte Carlo. Copied over from 
    parent class except to add the mask argument
    """
    previous_step_size_assign = []

    with tf.control_dependencies(previous_step_size_assign):
      next_state, kernel_results = self._impl.one_step(
          current_state, previous_kernel_results, seed=seed, mask=mask)
      return next_state, kernel_results
  

class MaskedUncalibratedHamiltonianMonteCarlo(UncalibratedHamiltonianMonteCarlo):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use `HamiltonianMonteCarlo(...)`
  or `MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...))`.

  For more details on `UncalibratedHamiltonianMonteCarlo`, see
  `HamiltonianMonteCarlo`.
  """

  # Add a setter for the log probability function that allows masks to be updated
  # throughout sampling
#   @property
#   def target_log_prob_fn(self):
#     return super().target_log_prob_fn

#   @UncalibratedHamiltonianMonteCarlo.target_log_prob_fn.setter
#   def set_target_log_prob_fn(self, new_log_prob_fn):
#     self.target_log_prob_fn.fset(self, new_log_prob_fn)

  def one_step(self, current_state, previous_kernel_results, seed=None, mask=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'hmc', 'one_step')):
      if self._store_parameters_in_results:
        step_size = previous_kernel_results.step_size
        num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
      else:
        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps

      [
          current_state_parts,
          step_sizes,
          current_target_log_prob,
          current_target_log_prob_grad_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          step_size,
          previous_kernel_results.target_log_prob,
          previous_kernel_results.grads_target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped)

      seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
      seeds = list(samplers.split_seed(seed, n=len(current_state_parts)))
      seeds = distribute_lib.fold_in_axis_index(
          seeds, self.experimental_shard_axis_names)

      current_momentum_parts = []
      for part_seed, x in zip(seeds, current_state_parts):
        current_momentum_parts.append(
            samplers.normal(
                shape=ps.shape(x),
                dtype=self._momentum_dtype or dtype_util.base_dtype(x.dtype),
                seed=part_seed))

      # Mask the momentum variables
      if mask is not None:
        for i, momentum in enumerate(current_momentum_parts):
          current_momentum_parts[i] = jnp.multiply(momentum, mask)

      with jax.disable_jit():
        integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
            self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,
      ] = integrator(current_momentum_parts,
                     current_state_parts,
                     current_target_log_prob,
                     current_target_log_prob_grad_parts)
      if self.state_gradients_are_stopped:
        next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]
    
      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      independent_chain_ndims = ps.rank(current_target_log_prob)

      new_kernel_results = previous_kernel_results._replace(
          log_acceptance_correction=_compute_log_acceptance_correction(
              current_momentum_parts, next_momentum_parts,
              independent_chain_ndims,
              shard_axis_names=self.experimental_shard_axis_names),
          target_log_prob=next_target_log_prob,
          grads_target_log_prob=next_target_log_prob_grad_parts,
          initial_momentum=current_momentum_parts,
          final_momentum=next_momentum_parts,
          seed=seed,
      )

      return maybe_flatten(next_state_parts), new_kernel_results
    
############################# Copying over internal functions ##############
def _compute_log_acceptance_correction(current_momentums,
                                       proposed_momentums,
                                       independent_chain_ndims,
                                       shard_axis_names=None,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.

  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:

  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```

  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.

  Inserting this into the detailed balance equation implies:

  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```

  One definition of `a(x'|x)` which satisfies (*) is:

  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```

  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)

  We call the bracketed term the "acceptance correction".

  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Assuming a standard Gaussian distribution for momentums, the chain eventually
  converges to:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  ```

  Relating this back to Metropolis-Hastings parlance, for HMC we have:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```

  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:

  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [exp(-0.5 z**2) / exp(-0.5 z'**2)]
                       target_prob(x)
  ```

  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)

  Args:
    current_momentums: `Tensor` representing the value(s) of the current
      momentum(s) of the state (parts).
    proposed_momentums: `Tensor` representing the value(s) of the proposed
      momentum(s) of the state (parts).
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    def compute_sum_sq(v, shard_axes):
      sum_sq = tf.reduce_sum(v ** 2., axis=ps.range(
          independent_chain_ndims, ps.rank(v)))
      if shard_axes is not None:
        sum_sq = distribute_lib.psum(sum_sq, shard_axes)
      return sum_sq
    shard_axis_names = (shard_axis_names or ([None] * len(current_momentums)))
    current_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(current_momentums, shard_axis_names)])
    proposed_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(proposed_momentums, shard_axis_names)])
    return 0.5 * mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=target_log_prob.dtype, name='step_size')
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob,
      grads_target_log_prob,
  ]
