from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import pdb
import ml_collections

# To-do: model must accept new sample as a categorical random variable that is then translated to a weight
# Mixed discrete and continuous variables?
# Also a bit non-trivial: How to define the categorical distribution over the hard constrained ensemble?
# Could one define a langevin dynamics over [0, 1, 1+]? The dynamics transition type if they hit the boundary.
# Does Gibbs with Gradients (and in general, the whole locally balanced formalism) work when state changes are not independent?
# In a hard constrained ensemble, edges are moved, rather than flipped.

# Key point: The use of the gradient is an approximation of the locally balanced proposal function. In cases when the energy function
# is far from quadratic, then this may incur significant loss of efficiency (see GWG paper). This will require some experimentation. In particular, 
# there is a tradeoff between (a) soft-constraining (or an interpolation between soft-hard constraint) and taking essentially Hamming ball 2 steps 
# between the current state and the loss of efficiency incurred via poor approximation to the LB function. For example, one may be able to 
# cheaply calculate the perturbation to a Lyapunov or Riccati solution incurred by swapping the edge function.

# In particular, the difference between the two solutions is probably an equation of the same form of low rank --> ADI (?)

# This fact also implies that being differentiable is not a requirement. Rather, the question is how expensive is it to caluclate or approximate the LB 
# ratio? For e.g., it could be trivially calculated for motif constraints via a change stat approach

# Might be wortwhile to just start with DLMCf/DMALA as it actually makes sense... It can be thought of as an Euler approximant

class DoubleWell(abstractmodel.AbstractModel):

    def __init__(self, config:ml_collections.ConfigDict):

        self.dim = config.dim
        self.range = config.range
        self.temp = config.temp
        self.offset = config.offset

    def make_init_params(self, rnd):
        params = {}
        params['params'] = jnp.array([self.temp, self.offset])
        return params
    
    def get_init_samples(self, rng, num_samples):
        # Draw random vector with dim uniformly over [-range, range]
        x0 = jax.random.randint(rng, (num_samples,) + self.dim, -self.range, self.range)
        return x0

    def forward(self, params, x):
        # Assume x has shape (n_samples, dim)

        params = params['params']
        temp = params[0]
        offset = params[1]
        
        H = jnp.power(jnp.tensordot(x, x, axes=(1, 1)) - offset, 2)
        return H
    
    def map(self, x):
        # Fix this indexing
        return jnp.tile(jnp.arange(-self.range, self.range), (x.shape[0], 1))[:, ]
    
    def get_value_and_grad(self, params, x):
        # Map x to integer
        x = self.map(x)
        x = x.astype(jnp.float32) # int tensor is not differentiable
        def fn(z):
            H = self.forward(params, z)
            return jnp.sum(H)
        
        val, grad = jax.value_and_grad(fn)(x)
        return val, grad

def build_model(config):
    return DoubleWell(config)