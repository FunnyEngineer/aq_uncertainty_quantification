from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.linen import initializers
from typing import Callable, Sequence
from jax import lax

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
