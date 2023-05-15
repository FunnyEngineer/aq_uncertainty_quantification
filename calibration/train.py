import jax.numpy as jnp
import jax
import pdb

@jax.vmap
def mse_loss(preds, targets):
  return jnp.mean((preds.flatten() - targets)**2)

@jax.jit
def apply_model(state, x, y):
    def loss_fn(params):
        y_hat = state.apply_fn({'params': params}, x)         # make forward pass
        loss = mse_loss(y_hat, y).mean()
        return loss, y_hat
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, y_hat), grads = grad_fn(state.params)
    return state, loss, grads


def train_one_epoch(state, training_generator):
    train_loss = []
    for i, (x, y) in enumerate(training_generator):
        state, loss, grads = apply_model(state, x, y)

        train_loss.append(loss.item())

        state = state.apply_gradients(grads=grads)
    train_loss = jnp.mean(jnp.array(train_loss))
    return state, train_loss