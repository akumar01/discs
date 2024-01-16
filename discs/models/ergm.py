"""ERGM Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class ERGM(abstractmodel.AbstractModel):
    """Degree-corrected stochastic block model."""

    def __init__(self, config: ml_collections.ConfigDict):
        self.shape = config.shape
        #self.A0 = config.A0
        N = self.shape[0]
        
        g = jnp.zeros(N)   #g for group
        g = g.at[int(N/2):].set(1)  #2 groups for now

        self.num_group = len(jnp.unique(g))

        b = self.num_group
        
        #building the matrix containers for the block connections parameters
        num_w = b**2   #this is how many mask matrices we need for the w's
        num_w_bi = int(b*(b+1)/2)

        # for w
        w_mask = []
        for k in range(b):
            for l in range(b):
                w_mask.append(jnp.zeros((N,N)))
                for i in range(N):
                    for j in range(N):
                        if i != j:  
                            if g[i] == l and g[j] == k:
                                w_mask[-1] = w_mask[-1].at[i,j].set(1)


        bi_mask = []

        for k in range(b):
            for l in range(k+1):
                bi_mask.append(jnp.zeros((N,N)))
                for i in range(N):
                    for j in range(N):
                        if i != j:  
                            if g[i] == l and g[j] == k:
                                bi_mask[-1] = bi_mask[-1].at[i,j].set(1)
                                bi_mask[-1] = bi_mask[-1].at[j,i].set(1)

        self.w_masks = jnp.concatenate(w_mask).reshape(num_w,N,N) 
        self.bi_masks = jnp.concatenate(bi_mask).reshape(num_w_bi,N,N)  


    def make_init_params(self, rnd):
        params = {}
        # connectivity strength
        N = self.shape[0]
        b = self.num_group #len(np.unique(g))  

        N_pars = int(2*N) + 1*b**2 + int(b*(b+1)/2)  

        pars = jax.random.uniform(rnd, shape=(N_pars,))

        alpha = pars[:N] # model in-degree param
        beta = pars[N:2*N]
        omega = pars[2*N:2*N+b**2]
        omega_bi = pars[2*N+b**2:2*N+b**2+int(b*(b+1)/2)]
        #mu = pars[2*N+b**2+int(b*(b+1)/2):]  #block connection weights parameter  

        return alpha, beta, omega, omega_bi

    def get_init_samples(self, rnd, num_samples: int):
        
        A0 = jax.random.bernoulli(
            rnd,
            shape= self.shape,
        ).astype(jnp.int32)

        #A0 = self.A0.astype(jnp.int32)
        return A0
    
    def forward(self, params, A):
        #energy fucntion (NOT log likelihood) of the degree-corrected SBM with mutual connection constraints
        if len(A.shape) >2: # b/c some mysterious block of code is making chains
            A = A[0]
        N = self.shape[0]
        
        alpha = params[0]
        beta = params[1]
        omega = params[2]
        omega_bi = params[3]
        w_masks = self.w_masks
        bi_masks = self.bi_masks

        U = jnp.tensordot(omega,w_masks,axes=([0], [0])) #alpha.reshape(-1,1)+beta+
        V = -jnp.tensordot(omega,w_masks,axes=([0], [0])) + jnp.tensordot(omega_bi,bi_masks,axes=([0], [0]))/2

        #Q = (torch.exp(torch.tensordot(mu,w_masks,axes=([0], [0])))-torch.exp(-(s_max+1)*torch.tensordot(mu,w_masks,axes=([0], [0]))))/(1-torch.exp(torch.tensordot(mu,w_masks,axes=([0], [0]))))
        #R = torch.exp(-torch.multiply(torch.tensordot(mu,w_masks,axes=([0], [0])),S))

        # ignore weights
        #Q = torch.ones((N,N))
        #R = torch.ones((N,N))
        
        #consider_mutual = True
        #if consider_mutual:
            #C = jnp.log(2*jnp.ones((N,N)))
            #UV2 = jnp.concatenate((-U,-U-V,C)).reshape(3,N,N)  
            #logZ = jax.scipy.special.logsumexp(UV2,axis=0) 
        #else:
            #C = jnp.zeros((N,N))
            #UC = jnp.concatenate((-U,C)).reshape(2,N,N)  
            #logZ = jax.scipy.special.logsumexp(UC,axis=0) 

        H = -jnp.multiply(A,U) - jnp.multiply(jnp.multiply(A,A.T), V)
          #- torch.multiply(torch.tensordot(mu,w_masks,axes=([0], [0])),S)- logZ  
        
        off_diag_mask = jnp.ones((N,N))-jnp.eye(N)    
        L = jnp.sum(jnp.multiply(H, off_diag_mask))
        #print(L)
        return L


    def get_value_and_grad(self, params, x):
        x = x.astype(jnp.float32)  # int tensor is not differentiable

        def fun(z):
            loglikelihood = self.forward(params, z)
            return jnp.sum(loglikelihood), loglikelihood

        (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
        return loglikelihood, grad


def build_model(config):
    return ERGM(config.model)
