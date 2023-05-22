from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.linen import initializers
from typing import Callable, Sequence
from jax import lax

import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

class Model(nn.Module):
    features: int
    kernel_init: Callable = initializers.lecun_normal()
    bias_init: Callable = initializers.zeros_init()
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features))
        y = lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            y = y + bias
        return y


class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

class MDN(nn.Module):
  num_components: int

  @nn.compact
  def __call__(self, x):
    x = nn.relu(nn.Dense(128)(x))
    x = nn.relu(nn.Dense(128)(x))
    x = nn.tanh(nn.Dense(64)(x))

    # Instead of regressing directly the value of the mass, the network
    # will now try to estimate the parameters of a mass distribution.
    # categorical_logits = nn.Dense(self.num_components)(x)
    # alpha = 1 +  nn.softplus(nn.Dense(self.num_components)(x))
    # beta = 1 + nn.softplus(nn.Dense(self.num_components)(x))
    mu = nn.Dense(1)(x)
    sigma = nn.relu(nn.Dense(1)(x))

    dist = tfd.Normal(loc= mu, scale = sigma)
    
    # dist = tfd.Independent(tfd.MixtureSameFamily( # gaussian distribution
    #     mixture_distribution=tfd.Categorical(logits=categorical_logits), 
    #     components_distribution=tfd.Beta(alpha, beta))) # normal funciton
    
    # dist = tfd.Independent(dist)
    return dist